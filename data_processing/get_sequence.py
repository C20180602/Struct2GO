'''
通过读取pdb文件, 输出蛋白质序列的onehot表示 (已停用 useless)
'''
# import imp
# from unicodedata import name
# import numpy as np
# import pandas as pd 
# from Bio import SeqIO
# from Bio.PDB.PDBParser import PDBParser
# import os
# import pickle

# def _load_sequence(filename):
#     if filename.endswith('.pdb'):
#         seq = load_predicted_PDB(filename)
#     S = seq2onehot(seq)
#     return S, seq


# def load_predicted_PDB(pdbfile):
#     # sequence from atom lines
#     records = SeqIO.parse(pdbfile, 'pdb-atom')
#     seqs = [str(r.seq) for r in records]
#     return seqs[0]

# def seq2onehot(seq):
#     """Create 26-dim embedding"""
#     chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
#              'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
#     vocab_size = len(chars)
#     vocab_embed = dict(zip(chars, range(vocab_size)))

#     # Convert vocab to one-hot
#     vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
#     for _, val in vocab_embed.items():
#         vocab_one_hot[val, val] = 1

#     embed_x = [vocab_embed[v] for v in seq]
#     seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

#     return seqs_x


# df=pd.read_csv("../data/protein_list.csv",sep=" ")
# list1=df.values.tolist()
# protein_list = np.array(list1)

# protein_node2one_hot = {}
# for file_name in os.listdir("/e/chensq/dag-classify/raw_data/HUMAN/pdb"):  
#     filename = file_name.split('-')[1]
#     if(filename in protein_list):
#         S, seqres = _load_sequence(os.path.join("/e/chensq/dag-classify/raw_data/HUMAN/pdb", file_name))
#         protein_node2one_hot[filename] = S
#         print(filename,S.shape)
# with open('../processed_data/protein_node2onehot','wb')as f:
#     pickle.dump(protein_node2one_hot,f)