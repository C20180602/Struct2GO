'''
根据论文, 这里是读取SeqVec特征, 输出到dict_sequence_feature文件中

ps: 失传多年的数据处理文件, 在清洗完服务器之后再也没被找到, 偶然在备份的U盘中发现了
'''
import pickle
import pandas as pd
import numpy as np

# 代代相传的序列mean特征，从HNetGO上找到了 (2333
with open('../data/9606-avg-emb.pkl','rb')as f:
    sequence_feature = pickle.load(f) 

df=pd.read_csv("../data/protein_list.csv",sep=" ")
list0=df.values.tolist()
protein_list = np.array(list0)

dict_sequence_feature={}
list1 = []
for i in list0:
    list1.append(i[0])

for name in list1:
    dict_sequence_feature[name] = [0.0]*1024

cnt=0
for protein in sequence_feature.keys():
    if protein in protein_list:
        dict_sequence_feature[protein] = sequence_feature[protein]
        cnt=cnt+1

with open('../processed_data/dict_sequence_feature','wb')as f:
    pickle.dump(dict_sequence_feature,f)
