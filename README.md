# ChEMBL Multitask Nerual Network model

- Based on the blogpost: http://chembl.blogspot.com/2019/05/multi-task-neural-network-on-chembl.html
- Model available in KNIME thanks to Greg Landrum: https://www.knime.com/blog/interactive-bioactivity-prediction-with-multitask-neural-networks

The model is exported to the ONNX format so it can be used in any programming language able to generate fingerprints with RDKit

# Example to predict in Python using the ONNX Runtime

```Python
import onnxruntime
import numpy as np
from rdkit import Chem
import json

FP_SIZE = 1024
RADIUS = 2

def calc_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, RADIUS, nBits=FP_SIZE)
    a = np.zeros((0,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, a)
    return a

def format_preds(preds):
    preds = np.concatenate(preds).ravel()
    np_preds = [(tar, pre) for tar, pre in zip(targets, preds)]
    dt = [('chembl_id','|U20'), ('pred', '<f4')]
    np_preds = np.array(np_preds, dtype=dt)
    np_preds[::-1].sort(order='pred')
    return np_preds

with open('targets_28.json') as json_file:
    targets = json.load(json_file)

# load the model
ort_session = onnxruntime.InferenceSession("chembl_28_multitask.onnx")

# calculate the FPs
smiles = 'CN(C)CCc1c[nH]c2ccc(C[C@H]3COC(=O)N3)cc12'
descs = calc_morgan_fp(smiles)

# run the prediction
ort_inputs = {ort_session.get_inputs()[0].name: descs}
preds = ort_session.run(None, ort_inputs)

# example of how the output of the model can be formatted
preds = format_preds(preds)
preds

array([('CHEMBL1163101', 0.97411716), ('CHEMBL1983', 0.9567785 ),
       ('CHEMBL4068', 0.9555353 ), ('CHEMBL4427', 0.9475912 ),
       ('CHEMBL2954', 0.9474703 ), ('CHEMBL4303', 0.93399465),
       ('CHEMBL3959', 0.93018764), ('CHEMBL5441', 0.9279458 ),
       ('CHEMBL313', 0.92713475), ('CHEMBL4296', 0.91438127),
       ('CHEMBL4616', 0.90646756), ('CHEMBL6141', 0.9001103 ),
       ('CHEMBL4561', 0.8965951 ), ('CHEMBL1980', 0.8889245 ),
       ...
```