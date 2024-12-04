# 复现流程(2024.12)

## 环境与数据
复现环境为 CUDA 12.1 ，torch 2.2.1+cu121 ，dgl 2.4.0+cu121 ，transformers 4.46.3

使用的人类蛋白质数据集，PDB文件来自AlphaFold数据库，node2vec特征来自struct_feature网盘数据，seqvec特征来自于HNetGO然后使用read_seqvec_features进行处理。
## 进入data_processing文件夹: 
1、python get_protein_list.py

2、python read_seqvec_feature.py

3、python read_node2vec_feature.py

4、python predicted_protein_struct2map.py

## 进入model文件夹：
python labels_load.py

结果记录：(2024.12.3)
```shell
--------------1: go term processing
--------------2: propagate labels
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20433/20433 [00:01<00:00, 11453.02it/s]
protein_num: 20433 -> 19653
--------------3: split protein set
20504
--------------4: read all kinds of features
protein_node2vec: 20504
seqvec_feature_dic: 20504
graph_dic: 20504
--------------5: process all kinds of files
now processing:  bp
----step 1----
total_process bp final_go_term_size 689
----step 2----
----step 3----
----step 4----
----step 5----
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17592/17592 [00:45<00:00, 386.95it/s]
bp dataset size: 17559
----step 6----
now processing:  cc
----step 1----
total_process cc final_go_term_size 284
----step 2----
----step 3----
----step 4----
----step 5----
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 18732/18732 [00:43<00:00, 429.23it/s]
cc dataset size: 18698
----step 6----
now processing:  mf
----step 1----
total_process mf final_go_term_size 328
----step 2----
----step 3----
----step 4----
----step 5----
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 18182/18182 [00:41<00:00, 442.68it/s]
mf dataset size: 18150
----step 6----
```

bp标签：689个
cc标签：284个
mf标签：328个

## 返回data_processing文件夹
python divide_data.py

运行结果(2024.12.3)
```shell
divide bp dataset
train dataset size 12291
valid dataset size 3511
test dataset size 1757
divide mf dataset
train dataset size 12705
valid dataset size 3630
test dataset size 1815
divide cc dataset
train dataset size 13088
valid dataset size 3739
test dataset size 1871
```

完成数据处理后，整个项目大小约为48G，divided_data占31G，processed_data占13G（使用人类蛋白质数据）

## 返回项目文件夹，开始训练
这里的-labels_num选项需要根据数据处理的结果进行填写

python train_Struct2GO.py -labels_num 328 -branch 'mf'

python train_Struct2GO.py -labels_num 689 -branch 'bp'

python train_Struct2GO.py -labels_num 284 -branch 'cc'

参数：

节点特征只用了node2vec没加onehot，可能稍差

batch_size=64

lr=1e-4

dropout=0.5

验证集结果：

mf: fmax:0.3822  auc:0.8239  aupr:0.4247  thresh:0.57

bp: fmax:0.3254  auc:0.7638  aupr:0.3157  thresh:0.54

cc: fmax:0.3948  auc:0.8245  aupr:0.4063  thresh:0.75


python eval_Struct2GO.py -branch 'mf' -thresh 0.57

python eval_Struct2GO.py -branch 'bp' -thresh 0.54

python eval_Struct2GO.py -branch 'cc' -thresh 0.75

测试集结果：

mf: fmax:0.3777  auc:0.8230  aupr:0.4322  thresh:0.57

bp: fmax:0.3282  auc:0.7634  aupr:0.3249  thresh:0.54

cc: fmax:0.3966  auc:0.8220  aupr:0.4167  thresh:0.75