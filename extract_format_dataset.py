import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sqlalchemy import create_engine
from sqlalchemy.sql import text
import tables as tb
from tables.atom import ObjectAtom

def parse_args():
    parser = argparse.ArgumentParser(description="Process ChEMBL data for multitask learning.")
    parser.add_argument(
        "--chembl_version", type=int, required=True, help="Version of ChEMBL database. (Required)"
    )
    parser.add_argument(
        "--fp_size", type=int, default=1024, help="Size of the fingerprints. Default: 1024"
    )
    parser.add_argument(
        "--radius", type=int, default=2, help="Radius for Morgan fingerprints. Default: 2"
    )
    parser.add_argument(
        "--active_mols", type=int, default=100, help="Minimum number of active molecules per target. Default: 100"
    )
    parser.add_argument(
        "--inactive_mols", type=int, default=100, help="Minimum number of inactive molecules per target. Default: 100"
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory to save output files. Default: current directory"
    )
    return parser.parse_args()

def fetch_data(engine):
    """
    Fetch relevant biological activity data from the database.

    This function retrieves data linking molecules, their targets, and activity metrics
    from a ChEMBL-like database. The resulting DataFrame contains information about
    compounds, targets, and activity values filtered for high-quality data.

    Args:
        engine: A SQLAlchemy database engine to connect to the database.

    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - doc_id: The document ID for the activity data.
            - standard_value: The activity measurement value (e.g., IC50, EC50) in nanomolar (nM) units.
            - molregno: The unique ChEMBL molecule registry number.
            - canonical_smiles: The canonical SMILES representation of the molecule.
            - tid: The target ID.
            - target_chembl_id: The ChEMBL ID for the target protein.
            - protein_class_desc: A description of the protein class for the target.
    """
    # SQL query to fetch relevant activity and target data
    query = """
    SELECT
      activities.doc_id AS doc_id,
      activities.standard_value AS standard_value,
      molecule_hierarchy.parent_molregno AS molregno,
      compound_structures.canonical_smiles AS canonical_smiles,
      target_dictionary.tid AS tid,
      target_dictionary.chembl_id AS target_chembl_id,
      protein_classification.protein_class_desc AS protein_class_desc
    FROM activities
      -- Join activities with assay, molecule, and target metadata tables
      JOIN assays ON activities.assay_id = assays.assay_id
      JOIN target_dictionary ON assays.tid = target_dictionary.tid
      JOIN target_components ON target_dictionary.tid = target_components.tid
      JOIN component_class ON target_components.component_id = component_class.component_id
      JOIN protein_classification ON component_class.protein_class_id = protein_classification.protein_class_id
      JOIN molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
      JOIN molecule_hierarchy ON molecule_dictionary.molregno = molecule_hierarchy.molregno
      JOIN compound_structures ON molecule_hierarchy.parent_molregno = compound_structures.molregno
    WHERE 
      -- Filter for activity measurements in nanomolar (nM) units
      activities.standard_units = 'nM' AND
      
      -- Only include specific types of activity measurements
      activities.standard_type IN ('EC50', 'IC50', 'Ki', 'Kd', 'XC50', 'AC50', 'Potency') AND

      -- Ensure data validity (no comments, duplicates, or uncertain data)
      activities.data_validity_comment IS NULL AND
      activities.standard_relation IN ('=', '<') AND
      activities.potential_duplicate = 0 AND

      -- Include assays with a high confidence score (>= 8)
      assays.confidence_score >= 8 AND

      -- Restrict to single-protein targets
      target_dictionary.target_type = 'SINGLE PROTEIN'
    """

    # Execute the SQL query using the provided database engine
    with engine.connect() as conn:
        # Fetch data into a pandas DataFrame using pyarrow for efficient type handling
        df = pd.read_sql(text(query), conn, dtype_backend="pyarrow")

    # Sort data by activity value (standard_value), molecule (molregno), and target (tid)
    df = df.sort_values(by=["standard_value", "molregno", "tid"], ascending=True)

    # Remove duplicate entries for each molecule-target pair, keeping the entry with the lowest standard_value
    df = df.drop_duplicates(subset=["molregno", "tid"], keep="first")

    return df

def set_active(row):
    """
    Determine if a molecule is active based on activity thresholds for protein families.
    Uses IDG protein family activity thresholds:
        - Kinases: <= 30 nM
        - GPCRs (G-Protein-Coupled Receptors): <= 100 nM
        - Nuclear Receptors: <= 100 nM
        - Ion Channels: <= 10 μM
        - Non-IDG Family Targets: <= 1 μM
    See: https://druggablegenome.net/IDGProteinFamilies

    Args:
        row (pd.Series): A row from a DataFrame containing 'standard_value' and 'protein_class_desc'.

    Returns:
        int: 1 if active, 0 if inactive.
    """
    standard_value = row["standard_value"]
    protein_class = row["protein_class_desc"]
    active = 0  # Default to inactive

    if standard_value is not pd.NA:
        # General threshold for activity
        if standard_value <= 1000:
            active = 1
        
        # Additional thresholds for specific protein families
        if "ion channel" in protein_class and standard_value <= 10000:
            active = 1
        if "enzyme  kinase  protein kinase" in protein_class and standard_value > 30:
            active = 0
        if "transcription factor  nuclear receptor" in protein_class and standard_value > 100:
            active = 0
        if "membrane receptor  7tm" in protein_class and standard_value > 100:
            active = 0
    
    return active

def filter_targets(df, active_mols, inactive_mols):
    """
    Filter targets based on activity and occurrence criteria.

    The filtering steps:
        1. Calculate activity for each molecule using `set_active`.
        2. Keep targets with at least `active_mols` active molecules.
        3. Keep targets with at least `inactive_mols` inactive molecules.
        4. Ensure targets appear in at least 2 different documents.

    Args:
        df (pd.DataFrame): DataFrame containing molecule and target data.
        active_mols (int): Minimum number of active molecules required for a target.
        inactive_mols (int): Minimum number of inactive molecules required for a target.

    Returns:
        Tuple[pd.DataFrame, set]:
            - Filtered DataFrame containing targets that meet all criteria.
            - Set of target ChEMBL IDs that pass the filters.
    """
    # Determine activity for all rows
    df["active"] = df.apply(set_active, axis=1)

    # Filter targets with enough active molecules
    acts = df[df["active"] == 1].groupby(["target_chembl_id"]).agg("count")
    acts = acts[acts["molregno"] >= active_mols].reset_index()["target_chembl_id"]

    # Filter targets with enough inactive molecules
    inacts = df[df["active"] == 0].groupby(["target_chembl_id"]).agg("count")
    inacts = inacts[inacts["molregno"] >= inactive_mols].reset_index()["target_chembl_id"]

    # Filter targets appearing in at least two different documents
    docs = df.drop_duplicates(subset=["doc_id", "target_chembl_id"])
    docs = docs.groupby(["target_chembl_id"]).agg("count")
    docs = docs[docs["doc_id"] >= 2.0].reset_index()["target_chembl_id"]

    # Intersect all criteria to get the final set of targets
    t_keep = set(acts).intersection(set(inacts)).intersection(set(docs))

    # Return the filtered DataFrame and the set of target ChEMBL IDs
    return df[df["target_chembl_id"].isin(t_keep)], t_keep

def calc_fp(smiles, mfpgen):
    """
    Calculate the molecular fingerprint for a given SMILES string.

    Args:
        smiles (str): The SMILES representation of the molecule.
        mfpgen: An RDKit fingerprint generator object.

    Returns:
        np.ndarray: A NumPy array representing the molecular fingerprint.
    """
    # Convert the SMILES string into an RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)

    # Generate the fingerprint for the molecule
    fp = mfpgen.GetFingerprint(mol)

    # Convert the fingerprint to a NumPy array
    a = np.zeros((0,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, a)

    return a


def save_to_h5(mt_df, descs, output_file):
    """
    Save molecular data, fingerprints, and labels into an HDF5 file.

    This function stores:
        - Molecule IDs (molregno)
        - Molecular fingerprints
        - Target ChEMBL IDs
        - Label matrix
        - Task weights for multi-task training

    Args:
        mt_df (pd.DataFrame): A DataFrame containing molecule data, labels, and target ChEMBL IDs.
        descs (np.ndarray): Array of molecular fingerprints.
        output_file (str): Path to the output HDF5 file.

    Returns:
        None
    """
    with tb.open_file(output_file, mode="w") as t_file:
        # Compression filters for efficient storage
        filters = tb.Filters(complib="blosc", complevel=5)

        # Save molecule IDs (molregno) as a variable-length array
        tatom = ObjectAtom()
        cids = t_file.create_vlarray(t_file.root, "molregnos", atom=tatom)
        for cid in mt_df["molregno"].values:
            cids.append(cid)

        # Save molecular fingerprints (fps) as a compressed array
        fatom = tb.Atom.from_dtype(descs.dtype)
        fps = t_file.create_carray(t_file.root, "fps", fatom, descs.shape, filters=filters)
        fps[:] = descs

        # Remove unused columns from the DataFrame (molregno and canonical_smiles)
        del mt_df["molregno"]
        del mt_df["canonical_smiles"]

        # Save target ChEMBL IDs as a variable-length array
        tatom = ObjectAtom()
        tcids = t_file.create_vlarray(t_file.root, "target_chembl_ids", atom=tatom)
        for tcid in mt_df.columns.values:
            tcids.append(tcid)

        # Save label matrix (task labels) as a compressed array
        labs = t_file.create_carray(t_file.root, "labels", fatom, mt_df.values.shape, filters=filters)
        labs[:] = mt_df.values

        # Calculate and save task weights
        # Each task's loss will be weighted inversely proportional to the number of data points for that task
        # Reference: http://www.bioinf.at/publications/2014/NIPS2014a.pdf
        weights = [1 / mt_df[mt_df[col] >= 0.0].shape[0] for col in mt_df.columns.values]
        weights = np.array(weights)
        ws = t_file.create_carray(t_file.root, "weights", fatom, weights.shape)
        ws[:] = weights

if __name__ == "__main__":
    args = parse_args()

    engine = create_engine(f"sqlite:///chembl_{args.chembl_version}.db")

    df = fetch_data(engine)
    df.to_csv(f"{args.output_dir}/chembl_{args.chembl_version}_activity_data.csv", index=False)

    activities, t_keep = filter_targets(df, args.active_mols, args.inactive_mols)
    activities.to_csv(f"{args.output_dir}/chembl_{args.chembl_version}_activity_data_filtered.csv", index=False)

    mt_df = activities.pivot(index="molregno", columns="target_chembl_id", values="active")
    mt_df = mt_df.fillna(-1).reset_index()

    structs = activities[["molregno", "canonical_smiles"]].drop_duplicates(subset="molregno")
    # keep only compounds that RDKit can parse
    structs = structs[structs["canonical_smiles"].apply(lambda smi: Chem.MolFromSmiles(smi) is not None)]

    mt_df = pd.merge(structs, mt_df, how="inner", on="molregno")
    mt_df.to_csv(f"{args.output_dir}/chembl_{args.chembl_version}_multi_task_data.csv", index=False)

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=args.radius,fpSize=args.fp_size)
    descs = np.asarray([calc_fp(smi, mfpgen) for smi in mt_df["canonical_smiles"]], dtype=np.float32)

    save_to_h5(mt_df, descs, f"{args.output_dir}/mt_data_{args.chembl_version}.h5")
