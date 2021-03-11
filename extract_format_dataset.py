from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sqlalchemy import create_engine
from sqlalchemy.sql import text
import numpy as np
import pandas as pd
import tables as tb
from tables.atom import ObjectAtom
import json


CHEMBL_VERSION = 28
FP_SIZE = 1024
RADIUS = 2
# num of active and inactive different molecules needed to include a target
ACTIVE_MOLS = 100
INACTIVE_MOLS = 100

# ----------------------------------------------------------------------------
# Get data from ChEMBL using the SQLite dump
engine = create_engine(f"sqlite:///chembl_{CHEMBL_VERSION}.db")

qtext = """
SELECT
  activities.doc_id                    AS doc_id,
  activities.standard_value            AS standard_value,
  molecule_hierarchy.parent_molregno   AS molregno,
  compound_structures.canonical_smiles AS canonical_smiles,
  molecule_dictionary.chembl_id        AS chembl_id,
  target_dictionary.tid                AS tid,
  target_dictionary.chembl_id          AS target_chembl_id,
  protein_family_classification.l1     AS l1,
  protein_family_classification.l2     AS l2,
  protein_family_classification.l3     AS l3
FROM activities
  JOIN assays ON activities.assay_id = assays.assay_id
  JOIN target_dictionary ON assays.tid = target_dictionary.tid
  JOIN target_components ON target_dictionary.tid = target_components.tid
  JOIN component_class ON target_components.component_id = component_class.component_id
  JOIN protein_family_classification ON component_class.protein_class_id = protein_family_classification.protein_class_id
  JOIN molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
  JOIN molecule_hierarchy ON molecule_dictionary.molregno = molecule_hierarchy.molregno
  JOIN compound_structures ON molecule_hierarchy.parent_molregno = compound_structures.molregno
WHERE activities.standard_units = 'nM' AND
      activities.standard_type IN ('EC50', 'IC50', 'Ki', 'Kd', 'XC50', 'AC50', 'Potency') AND
      activities.data_validity_comment IS NULL AND
      activities.standard_relation IN ('=', '<') AND
      activities.potential_duplicate = 0 AND assays.confidence_score >= 8 AND
      target_dictionary.target_type = 'SINGLE PROTEIN'"""

with engine.begin() as conn:
    res = conn.execute(text(qtext))
    df = pd.DataFrame(res.fetchall())

df.columns = res.keys()
df = df.where((pd.notnull(df)), None)

# Drop duplicate activities keeping the activity with lower concentration for each molecule-target pair
df = df.sort_values(by=["standard_value", "molregno", "tid"], ascending=True)
df = df.drop_duplicates(subset=["molregno", "tid"], keep="first")

# save to csv
df.to_csv(f"chembl_{CHEMBL_VERSION}_activity_data.csv", index=False)


# ----------------------------------------------------------------------------
# Set to active/inactive by threshold
#     Based on IDG protein families: https://druggablegenome.net/ProteinFam
#         Kinases: <= 30nM
#         GPCRs: <= 100nM
#         Nuclear Receptors: <= 100nM
#         Ion Channels: <= 10μM
#         Non-IDG Family Targets: <= 1μM
def set_active(row):
    active = 0
    if row["standard_value"] <= 1000:
        active = 1
    if row["l1"] == "Ion channel":
        if row["standard_value"] <= 10000:
            active = 1
    if row["l2"] == "Kinase":
        if row["standard_value"] > 30:
            active = 0
    if row["l2"] == "Nuclear receptor":
        if row["standard_value"] > 100:
            active = 0
    if row["l3"] and "GPCR" in row["l3"]:
        if row["standard_value"] > 100:
            active = 0
    return active


df["active"] = df.apply(lambda row: set_active(row), axis=1)


# ----------------------------------------------------------------------------
# Filter target data
#     Keep targets mentioned at least in two different docs
#     Keep targets with at least ACTIVE_MOLS active and INACTIVE_MOLS inactive molecules.

# get targets with at least ACTIVE_MOLS different active molecules
acts = df[df["active"] == 1].groupby(["target_chembl_id"]).agg("count")
acts = acts[acts["molregno"] >= ACTIVE_MOLS].reset_index()["target_chembl_id"]

# get targets with at least INACTIVE_MOLS different inactive molecules
inacts = df[df["active"] == 0].groupby(["target_chembl_id"]).agg("count")
inacts = inacts[inacts["molregno"] >= INACTIVE_MOLS].reset_index()["target_chembl_id"]

# get targets mentioned in at least two docs
docs = df.drop_duplicates(subset=["doc_id", "target_chembl_id"])
docs = docs.groupby(["target_chembl_id"]).agg("count")
docs = docs[docs["doc_id"] >= 2.0].reset_index()["target_chembl_id"]

# keep data for filtered targets
t_keep = set(acts).intersection(set(inacts)).intersection(set(docs))
activities = df[df["target_chembl_id"].isin(t_keep)]

ion = pd.unique(activities[activities["l1"] == "Ion channel"]["tid"]).shape[0]
kin = pd.unique(activities[activities["l2"] == "Kinase"]["tid"]).shape[0]
nuc = pd.unique(activities[activities["l2"] == "Nuclear receptor"]["tid"]).shape[0]
gpcr = pd.unique(
    activities[activities["l3"].str.contains("GPCR", na=False)]["tid"]
).shape[0]

print("Number of unique targets: ", len(t_keep))
print("  Ion channel: ", ion)
print("  Kinase: ", kin)
print("  Nuclear receptor: ", nuc)
print("  GPCR: ", gpcr)
print("  Others: ", len(t_keep) - ion - kin - nuc - gpcr)

# save it to a file
activities.to_csv(f"chembl_{CHEMBL_VERSION}_activity_data_filtered.csv", index=False)


# ----------------------------------------------------------------------------
# Prepare the label matrix for the multi-task deep neural network
#     known active = 1
#     known no-active = 0
#     unknown activity = -1, so we'll be able to easilly filter them and won't be taken into account when calculating the loss during model training.

# The matrix is extremely sparse so using sparse matrices (COO/CSR/CSC) should be considered. There are a couple of issues making it a bit tricker than what it should be so we'll keep the example without them.
#     https://github.com/pytorch/pytorch/issues/20248
#     https://github.com/scipy/scipy/issues/7531


def gen_dict(group):
    return {tid: act for tid, act in zip(group["target_chembl_id"], group["active"])}


group = activities.groupby("chembl_id")
temp = pd.DataFrame(group.apply(gen_dict))
mt_df = pd.DataFrame(temp[0].tolist())
mt_df["chembl_id"] = temp.index
mt_df = mt_df.where((pd.notnull(mt_df)), -1)


structs = activities[["chembl_id", "canonical_smiles"]].drop_duplicates(
    subset="chembl_id"
)

# drop mols not sanitizing on rdkit
structs["romol"] = structs.apply(
    lambda row: Chem.MolFromSmiles(row["canonical_smiles"]), axis=1
)
structs = structs.dropna()
del structs["romol"]

# add the structures to the final df
mt_df = pd.merge(structs, mt_df, how="inner", on="chembl_id")

# save df to csv
mt_df.to_csv(f"chembl_{CHEMBL_VERSION}_multi_task_data.csv", index=False)


# ----------------------------------------------------------------------------
# Calc fingeprints and save data to a PyTables H5 file
def calc_fp(smiles, fp_size, radius):
    """
    calcs morgan fingerprints as a numpy array.
    """
    mol = Chem.MolFromSmiles(smiles)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)
    a = np.zeros((0,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, a)
    return a


# calc fps
descs = [calc_fp(smi, FP_SIZE, RADIUS) for smi in mt_df["canonical_smiles"].values]
descs = np.asarray(descs, dtype=np.float32)

# store all training data in a PyTables file
with tb.open_file(f"mt_data_{CHEMBL_VERSION}.h5", mode="w") as t_file:

    # set compression filter. It will make the file much smaller
    filters = tb.Filters(complib="blosc", complevel=5)

    # save chembl_ids
    tatom = ObjectAtom()
    cids = t_file.create_vlarray(t_file.root, "chembl_ids", atom=tatom)
    for cid in mt_df["chembl_id"].values:
        cids.append(cid)

    # save fps
    fatom = tb.Atom.from_dtype(descs.dtype)
    fps = t_file.create_carray(t_file.root, "fps", fatom, descs.shape, filters=filters)
    fps[:] = descs

    del mt_df["chembl_id"]
    del mt_df["canonical_smiles"]

    # save target chembl ids
    tcids = t_file.create_vlarray(t_file.root, "target_chembl_ids", atom=tatom)
    for tcid in mt_df.columns.values:
        tcids.append(tcid)

    # save labels
    labs = t_file.create_carray(
        t_file.root, "labels", fatom, mt_df.values.shape, filters=filters
    )
    labs[:] = mt_df.values

    # save task weights
    # each task loss will be weighted inversely proportional to its number of data points
    # from: http://www.bioinf.at/publications/2014/NIPS2014a.pdf
    weights = []
    for col in mt_df.columns.values:
        c = mt_df[mt_df[col] >= 0.0].shape[0]
        weights.append(1 / c)
    weights = np.array(weights)
    ws = t_file.create_carray(t_file.root, "weights", fatom, weights.shape)
    ws[:] = weights


# Open H5 file and show the shape of all collections
with tb.open_file(f"mt_data_{CHEMBL_VERSION}.h5", mode="r") as t_file:
    print(t_file.root.chembl_ids.shape)
    print(t_file.root.target_chembl_ids.shape)
    print(t_file.root.fps.shape)
    print(t_file.root.labels.shape)
    print(t_file.root.weights.shape)

    # save targets to a json file
    with open(f"targets_{CHEMBL_VERSION}.json", "w") as f:
        json.dump(t_file.root.target_chembl_ids[:], f)
