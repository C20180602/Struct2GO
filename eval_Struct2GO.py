import torch
from torch import nn
import torch.nn.functional as F
import argparse
import numpy as np
import dgl
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import pickle
from data_processing.divide_data import MyDataSet
from model.evaluation import cacul_aupr,calculate_performance
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import warnings
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging
import json

def create_logger(branch_name):
    logger = logging.getLogger(branch_name)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=os.path.join('log','test_'+branch_name+'.log'))
    logger.setLevel(logging.DEBUG)
    handler1.setLevel(logging.ERROR)
    handler2.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

warnings.filterwarnings('ignore')

# TODO 个人认为，测试集不用再枚举thresh了，直接使用验证集得出的最优thresh即可
Thresholds = [x/100 for x in range(1,100)]

if __name__ == "__main__":
    
    device = "cuda:7"
    #参数设置
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-branch', '--branch',type=str,default='mf')
    parser.add_argument('-thresh', '--thresh',type=float,default=0.5)
    args = parser.parse_args()
    
    input_thresh = args.thresh
    test_data_path = 'divided_data/'+args.branch+'_test_dataset'
    label_network_path = 'processed_data/label_'+args.branch+'_network'
    term2idx_path = 'processed_data/'+args.branch+'_term2idx.json'
    model_path = 'save_models/bestmodel_'+args.branch+'_64_0.0001_0.5.pkl'
    
    logger = create_logger(args.branch)
    
    with open(test_data_path,'rb') as f:
        test_dataset = pickle.load(f)
    with open(label_network_path,'rb') as f:
        label_network = pickle.load(f)
    with open(term2idx_path,'r') as f:
        term2idx = json.load(f)
        idx2term = term2idx.keys()
    label_network = label_network.to(device)
    label_topo_order_list = dgl.topological_nodes_generator(label_network)
    model = torch.load(model_path)

    batch_size = 64
    test_dataloader = GraphDataLoader(dataset=test_dataset, batch_size = batch_size, drop_last = False, shuffle = False)
    criterion = nn.CrossEntropyLoss()
    logger.info('#########'+args.branch+'###########')
    logger.info('########start testing###########') 

    t_loss = 0
    test_batch_num = 0
    pred = []
    actual = []
    protein_list = []
    model.eval()
    print("testing")
    with torch.no_grad():
        for i, (pids, graphs, labels, seq_feats) in tqdm(enumerate(test_dataloader)):
            graphs = graphs.to(device)
            seq_feats = seq_feats.to(device)
            labels = labels.to(device)
            labels = torch.squeeze(labels)
            if len(labels.shape)==1:
                labels = labels.unsqueeze(0)
            
            logits = model(graphs,seq_feats,label_network)
            logits = F.sigmoid(logits)
            
            loss = criterion(logits,labels)
            
            protein_list += pids
            t_loss += loss.item()
            pred += logits.tolist()
            actual += labels.tolist()
    
    # 为了保持可控，这里使用传入的thresh来确定最终分类结果
    assert len(pred) == len(actual)
    assert len(pred) == len(protein_list)
    assert len(pred[0]) == len(idx2term)
    result = {}
    for i in range(len(pred)):
        protein = protein_list[i]
        result[protein] = []
        y_ = pred[i]
        y = actual[i]
        for j,x in enumerate(idx2term):
            # 寻找新预测出来的标签
            if y[j] < 1. and y_[j] > input_thresh:
                result[protein].append(x+f" {y_[j]:.5f}")
    with open('test_result/'+args.branch+'_result.json','w') as f:
        json.dump(result,f,indent=4)
        
    t_loss /= len(test_dataloader)
    fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)
    aupr = cacul_aupr(np.array(actual).flatten(), np.array(pred).flatten())

    each_best_fcore = 0
    each_best_scores = []
    for thresh in tqdm(Thresholds):
        f_score,precision, recall  = calculate_performance(actual, pred, label_network,threshold=thresh)
        if f_score >= each_best_fcore:
            each_best_fcore = f_score
            each_best_scores = [thresh, f_score, recall, precision]
    t, f_score, recall, precision = each_best_scores[0], each_best_scores[1], each_best_scores[2], each_best_scores[3]
    logger.info('loss: {}, thresh: {}, f_score {}'.format(t_loss, t, f_score))
    logger.info('auc {}, recall {}, precision {},aupr {}'.format(auc_score, recall, precision, aupr))
    print('loss: {}, thresh: {}, f_score {}'.format(t_loss, t, f_score))
    print('auc {}, recall {}, precision {},aupr {}'.format(auc_score, recall, precision, aupr))
