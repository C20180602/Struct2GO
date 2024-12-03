'''
读取PDB文件, 通过C-alpha原子间的距离小于10来建图
运行前需要删除data/proteins_edges文件夹
'''
from unicodedata import name
import numpy as np
import pandas as pd 
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import os
import warnings
from tqdm import tqdm

def _load_cmap(filename, cmap_thresh=10.0):
    if filename.endswith('.pdb'):
        D, seq = load_predicted_PDB(filename)
        A = np.double(D < cmap_thresh)
        #print(A)
    S = seq2onehot(seq)
    S = S.reshape(1, *S.shape)
    A = A.reshape(1, *A.shape)

    return A, S, seq


def load_predicted_PDB(pdbfile):
    # Generate (diagonalized) C_alpha distance matrix from a pdbfile
    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
    #name = structure.header['name']
    #print(name)
    residues = [r for r in structure.get_residues()]

    # sequence from atom lines
    records = SeqIO.parse(pdbfile, 'pdb-atom')
    seqs = [str(r.seq) for r in records]

    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one-two)

    return distances, seqs[0]

def seq2onehot(seq):
    """Create 26-dim embedding"""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed[v] for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x

# 忽略读取AlphaFold文件时没有PDB ID的警告
warnings.filterwarnings("ignore")
os.mkdir("../data/proteins_edges")
# 这里的文件路径并不重要，只需要将所有PDB文件放在一个文件夹下即可
for path,dir_list,file_list in os.walk("/e/chensq/dag-classify/raw_data/HUMAN/pdb"):  
    for file_name in tqdm(file_list):
        A, S, seqres = _load_cmap(os.path.join(path, file_name),cmap_thresh=10.0)
        B = np.reshape(A,(-1,len(A[0])))
        result = []
        N = len(B)
        for i in range(N):
            for j in range(N):
                tmp1 = []
                if B[i][j] and i!=j:
                    tmp1.append(i)
                    tmp1.append(j)
                    result.append(tmp1)
        np.array(result)
        #print(result)
        filename = file_name.split("-")
        name = filename[1]
        data = pd.DataFrame(result)
        #index参数设置为False表示不保存行索引,header设置为False表示不保存列索引
        data.to_csv("../data/proteins_edges/" + name + ".txt",sep=" ",index=False,header=False)
        #B_ = matrix2table(B)
        #print(len(A))
        #A_ = matrix2table()



    