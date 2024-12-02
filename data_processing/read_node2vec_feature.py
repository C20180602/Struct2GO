import numpy as np
import os
import pickle

protein_node2vec={}
for path,dir_list,file_list in os.walk("../data/struct_feature"):
    for file in file_list:
        protein_name = file.split('.')[0]
        print(protein_name)
        data = np.loadtxt((os.path.join(path, file)))
        protein_node2vec[protein_name] = data
        print(data.shape)
with open('../processed_data/protein_node2vec','wb')as f:
    pickle.dump(protein_node2vec,f)