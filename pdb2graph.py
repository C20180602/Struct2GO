'''
获取蛋白质结构图DGLGraph
图特征：
node: "feature": 20
'''
import Bio.PDB.PDBParser
import os
import numpy as np
import torch
import dgl
import warnings
import pickle
from tqdm import tqdm

parser = Bio.PDB.PDBParser()
thresh_aa = 10.0

three2one= {
    'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q',
    'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 
    'TYR':'Y', 'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 
    'MET':'M', 'ALA':'A', 'GLY':'G', 'PRO':'P', 'CYS':'C'
}
# 氨基酸的特征 20维
'''
AA_feature_onehot = {
    'A' : torch.tensor([ 1.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'R' : torch.tensor([ 0.,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'N' : torch.tensor([ 0.,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'D' : torch.tensor([ 0.,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'C' : torch.tensor([ 0.,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'Q' : torch.tensor([ 0.,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'E' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'G' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'H' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'I' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'L' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'K' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0]),
    'M' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0]),
    'F' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0]),
    'P' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0]),
    'S' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0]),
    'T' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0]),
    'W' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0]),
    'Y' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0]),
    'V' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]),
}
'''
AA_feature_blosum80 = {
    'A' : torch.tensor([ 7., -3, -3, -3, -1, -2, -2,  0, -3, -3, -3, -1, -2, -4, -1,  2,  0, -5, -4, -1]),
    'R' : torch.tensor([-3.,  9, -1, -3, -6,  1, -1, -4,  0, -5, -4,  3, -3, -5, -3, -2, -2, -5, -4, -4]),
    'N' : torch.tensor([-3., -1,  9,  2, -5,  0, -1, -1,  1, -6, -6,  0, -4, -6, -4,  1,  0, -7, -4, -5]),
    'D' : torch.tensor([-3., -3,  2, 10, -7, -1,  2, -3, -2, -7, -7, -2, -6, -6, -3, -1, -2, -8, -6, -6]),
    'C' : torch.tensor([-1., -6, -5, -7, 13, -5, -7, -6, -7, -2, -3, -6, -3, -4, -6, -2, -2, -5, -5, -2]),
    'Q' : torch.tensor([-2.,  1,  0, -1, -5,  9,  3, -4,  1, -5, -4,  2, -1, -5, -3, -1, -1, -4, -3, -4]),
    'E' : torch.tensor([-2., -1, -1,  2, -7,  3,  8, -4,  0, -6, -6,  1, -4, -6, -2, -1, -2, -6, -5, -4]),
    'G' : torch.tensor([ 0., -4, -1, -3, -6, -4, -4,  9, -4, -7, -7, -3, -5, -6, -5, -1, -3, -6, -6, -6]),
    'H' : torch.tensor([-3.,  0,  1, -2, -7,  1,  0, -4, 12, -6, -5, -1, -4, -2, -4, -2, -3, -4,  3, -5]),
    'I' : torch.tensor([-3., -5, -6, -7, -2, -5, -6, -7, -6,  7,  2, -5,  2, -1, -5, -4, -2, -5, -3,  4]),
    'L' : torch.tensor([-3., -4, -6, -7, -3, -4, -6, -7, -5,  2,  6, -4,  3,  0, -5, -4, -3, -4, -2,  1]),
    'K' : torch.tensor([-1.,  3,  0, -2, -6,  2,  1, -3, -1, -5, -4,  8, -3, -5, -2, -1, -1, -6, -4, -4]),
    'M' : torch.tensor([-2., -3, -4, -6, -3, -1, -4, -5, -4,  2,  3, -3,  9,  0, -4, -3, -1, -3, -3,  1]),
    'F' : torch.tensor([-4., -5, -6, -6, -4, -5, -6, -6, -2, -1,  0, -5,  0, 10, -6, -4, -4,  0,  4, -2]),
    'P' : torch.tensor([-1., -3, -4, -3, -6, -3, -2, -5, -4, -5, -5, -2, -4, -6, 12, -2, -3, -7, -6, -4]),
    'S' : torch.tensor([ 2., -2,  1, -1, -2, -1, -1, -1, -2, -4, -4, -1, -3, -4, -2,  7,  2, -6, -3, -3]),
    'T' : torch.tensor([ 0., -2,  0, -2, -2, -1, -2, -3, -3, -2, -3, -1, -1, -4, -3,  2,  8, -5, -3,  0]),
    'W' : torch.tensor([-5., -5, -7, -8, -5, -4, -6, -6, -4, -5, -4, -6, -3,  0, -7, -6, -5, 16,  3, -5]),
    'Y' : torch.tensor([-4., -4, -4, -6, -5, -3, -5, -6,  3, -3, -2, -4, -3,  4, -6, -3, -3,  3, 11, -3]),
    'V' : torch.tensor([-1., -4, -5, -6, -2, -4, -4, -6, -5,  4,  1, -4,  1, -2, -4, -3,  0, -5, -3,  7]),
}

# 将蛋白质中的每个氨基酸的Cα原子作为
def get_graph_info(pdbfilepath):
    struct = parser.get_structure(0,pdbfilepath)
    # 只取第一个模型
    model = struct.get_models().__next__()
    graph = dgl.DGLGraph()
    
    edge_cnt = 0
    ca_pos_list = []
    for chain in model.get_chains():
        residues = chain.get_residues()
        for residue in residues:
            res_name = three2one[residue.get_resname()]
            ca_pos_list.append(torch.from_numpy(residue['CA'].get_coord()))
            residue_feature = torch.cat([AA_feature_blosum80[res_name]],dim=0)
            graph.add_nodes(1, {'feature': residue_feature.unsqueeze(0)})
    
    n = len(ca_pos_list)
    for i in range(n):
        Cpos1 = ca_pos_list[i]
        ca_pos_tensor = torch.stack(ca_pos_list)
        dis_tensor = torch.norm(ca_pos_tensor-Cpos1,dim=1)
        j_tensor = torch.where(dis_tensor < thresh_aa)[0]
        cadidate_num = j_tensor.shape[0]
        graph.add_edges(torch.tensor([i]*cadidate_num), j_tensor)
        edge_cnt += cadidate_num
        
    print("edge cnt:",edge_cnt)
    return graph

if __name__ == "__main__":
    # 忽略使用DGLGraph建立空图的警告
    warnings.filterwarnings("ignore")
    pdb_graph_dict = {}
    pdb_data_dir = os.path.join("/e/protein/AF2DB")
    for pdb_file in tqdm(os.listdir(pdb_data_dir)):
        pid = pdb_file.split(".")[0]
        graph = get_graph_info(os.path.join(pdb_data_dir, pdb_file))
        pdb_graph_dict[pid] = graph

    # 使用pickle保存图数据
    with open("pdb_graph_dict.pkl", "wb") as f:
        pickle.dump(pdb_graph_dict, f)
    print("Graph data saved to af2db_graph_dict.pkl")
    print("Total proteins processed:", len(pdb_graph_dict))