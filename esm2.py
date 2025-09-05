'''
用于生成序列中每个氨基酸的特征向量
'''
import json
import os
import pickle

import torch
import esm
import Bio.PDB.PDBParser
from pathlib import Path
from tqdm import tqdm

three2one= {
    'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q',
    'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 
    'TYR':'Y', 'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 
    'MET':'M', 'ALA':'A', 'GLY':'G', 'PRO':'P', 'CYS':'C'
}

def get_seq_info(pdbfilepath):
    parser = Bio.PDB.PDBParser()
    struct = parser.get_structure(0,pdbfilepath)
    # 只取第一个模型
    model = struct.get_models().__next__()
    # i表示当前氨基酸的序号
    seq = []
    for res in model.get_residues():
        acid = three2one[res.get_resname()]
        seq.append(acid)
    return "".join(seq)

# 设置CUDA设备编号
device = "cuda:1"

# ESM预训练模型初始化,取消梯度
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
converter = alphabet.get_batch_converter()
model.to(device)
for p in model.parameters():
    p.requires_grad = False
model.eval()

seq_emb_dict = {}
pdb_dir_path = "/e/protein/AF2DB"
for pdb_file in tqdm(os.listdir(pdb_dir_path)):
    if not pdb_file.endswith(".pdb"):
        continue
    pid = pdb_file.split(".")[0]
    seq = get_seq_info(Path(pdb_dir_path,pdb_file))
    
    labels, strs, tokens = converter([(pid,seq)])
    tokens = tokens.to(device)
    seq_emb = model(tokens, repr_layers=[36])['representations'][36]
    seq_emb = seq_emb.squeeze(0).sum(dim=0)
    seq_emb = seq_emb.cpu().numpy()
    seq_emb_dict[pid] = seq_emb
    dir_path = Path("dataset",pid)
    del tokens,seq_emb
    torch.cuda.empty_cache()

with open("esm2_seq_emb.pkl","wb") as f:
    pickle.dump(seq_emb_dict,f)
print("序列特征向量生成完毕")