import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import json
import dgl
import torchinfo
from model.network import SAGNetworkHierarchical
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc
import queue

class MyDataset(Dataset):
    def __init__(self, X, G, y):
        self.X = X
        self.G = G
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.G[idx], self.y[idx]

def my_collate(batch):
    Xs, Gs, ys = zip(*batch)
    X = torch.stack(Xs).float()
    G = dgl.batch(Gs)
    y = torch.stack(ys).float()
    return X, G, y

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs):
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for X, G, labels in train_loader:
            X = X.to(device)
            G = G.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(G, X).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for X, G, labels in val_loader:
                X = X.to(device)
                G = G.to(device)
                labels = labels.to(device)
                
                outputs = model(G, X).squeeze()
                loss = criterion(outputs, labels)
                running_loss += loss.item() * X.size(0)
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # 更新学习率
        scheduler.step(epoch_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
    
    return train_losses, val_losses

def predict(model, test_loader):
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X, G, labels in test_loader:
            X = X.to(device)
            G = G.to(device)
            labels = labels.to(device)
            
            outputs = model(G, X).squeeze()
            probs = outputs.cpu().numpy()
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    return all_labels, all_probs

def cal_metrics(pred, actual, thresh=None):
    actual = np.array(actual).flatten()
    pred = np.array(pred).flatten()
    fpr, tpr, th = roc_curve(actual, pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    precision, recall, pr_thresh = metrics.precision_recall_curve(actual, pred)
    aupr_score = metrics.auc(recall, precision)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    if thresh is None:
        thresh = pr_thresh[np.argmax(f1)]
        fmax = np.max(f1)
    else:
        index = np.searchsorted(pr_thresh, thresh)
        fmax = f1[index]
    return auc_score, aupr_score, fmax, thresh

def propagate(go_list, go_dict, go_id_dict):
    vis={}
    q = queue.Queue()
    for go in go_list:
        if go in go_id_dict:
            q.put(go)
            vis[go]=1
    while not q.empty():
        u = q.get()
        go_info = go_dict[go_id_dict[u]]
        if "is_a" in go_info:
            for v in go_info['is_a']:
                if v not in vis:
                    vis[v]=1
                    q.put(v)
        if "part_of" in go_info:
            for v in go_info['part_of']:
                if v not in vis:
                    vis[v]=1
                    q.put(v)
    return vis.keys()

# 检查是否有可用的GPU
device = torch.device("cuda:1")

# 读取ESM嵌入特征
with open('esm2_seq_emb.pkl', 'rb') as f:
    esm_dict = pkl.load(f)
print("reading esm embedding done.")
    
# 读取PDB图数据
pdb_graph_dict = {}
with open('pdb_graph_dict.pkl', 'rb') as f:
    pdb_graph_dict = pkl.load(f)
print("reading graphs done.")

# 读取GO obo信息
with open('../go_terms.json', 'r') as f:
    go_dict = json.load(f)
with open('../go_id.json', 'r') as f:
    go_id_dict = json.load(f)
bp_list = []
cc_list = []
mf_list = []
for i, go_info in enumerate(go_dict):
    if go_info['namespace'] == 'biological_process':
        bp_list.append(go_info['id'])
    elif go_info['namespace'] == 'cellular_component':
        cc_list.append(go_info['id'])
    elif go_info['namespace'] == 'molecular_function':
        mf_list.append(go_info['id'])
bp_id_dict = {bp: i for i, bp in enumerate(bp_list)}
cc_id_dict = {cc: i for i, cc in enumerate(cc_list)}
mf_id_dict = {mf: i for i, mf in enumerate(mf_list)}

# 读取标签
pid_bp_label_dict = {}
pid_cc_label_dict = {}
pid_mf_label_dict = {}
pid_all_label_dict = {}
with open('../af2db_go.tsv', 'r') as f:
    f.readline()  # 跳过标题行
    while True:
        line = f.readline()
        if not line:
            break
        parts = line.strip().split('\t')
        pid = parts[0]
        if pid not in esm_dict:
            continue
        if pid not in pdb_graph_dict:
            continue
        if len(parts) < 3:
            continue
        go_terms = parts[2].split('; ')
        go_terms = propagate(go_terms, go_dict, go_id_dict)
        for go in go_terms:
            if go in bp_id_dict:
                if pid not in pid_bp_label_dict:
                    pid_bp_label_dict[pid] = np.zeros(len(bp_list))
                pid_bp_label_dict[pid][bp_id_dict[go]] = 1
            elif go in cc_id_dict:
                if pid not in pid_cc_label_dict:
                    pid_cc_label_dict[pid] = np.zeros(len(cc_list))
                pid_cc_label_dict[pid][cc_id_dict[go]] = 1
            elif go in mf_id_dict:
                if pid not in pid_mf_label_dict:
                    pid_mf_label_dict[pid] = np.zeros(len(mf_list))
                pid_mf_label_dict[pid][mf_id_dict[go]] = 1
                
            if go in go_id_dict:
                if pid not in pid_all_label_dict:
                    pid_all_label_dict[pid] = np.zeros(len(go_dict))
                pid_all_label_dict[pid][go_id_dict[go]] = 1
print("reading go labels done.")

# 划分数据集 8:1:1
for ont in ['bp', 'cc', 'mf', 'all']:
    if ont == 'bp':
        pid_label_dict = pid_bp_label_dict
    elif ont == 'cc':
        pid_label_dict = pid_cc_label_dict
    elif ont == 'mf':
        pid_label_dict = pid_mf_label_dict
    else:
        pid_label_dict = pid_all_label_dict

    X = []
    G = []
    y = []
    for pid, label in pid_label_dict.items():
        if pid in esm_dict:
            X.append(torch.tensor(esm_dict[pid]))
            G.append(pdb_graph_dict[pid])
            y.append(torch.tensor(label))
            
    n_samples = len(X)
    n_labels = y[0].shape[0]
    print(f"Ontology: {ont}")
    print(f"Total samples: {n_samples}, Features: 2560, Labels: {n_labels}")

    # 划分数据集
    X_temp, X_test, G_temp, G_test, y_temp, y_test = train_test_split(X, G, y, test_size=0.1)
    X_train, X_val, G_train, G_val, y_train, y_val = train_test_split(X_temp, G_temp, y_temp, test_size=0.1)

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # 转换为PyTorch张量并创建数据加载器
    train_dataset = MyDataset(X_train, G_train, y_train)
    val_dataset = MyDataset(X_val, G_val, y_val)
    test_dataset = MyDataset(X_test, G_test, y_test)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate)

    # 初始化模型、损失函数和优化器
    model = SAGNetworkHierarchical(20, 512, n_labels, 2560, num_convs=3, pool_ratio=0.4, dropout=0.5).to(device)
    with open(f'struct2go_{ont}_summary.txt', 'w') as f:
        f.write(torchinfo.summary(model).__str__())
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 训练模型
    print("开始训练模型...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs=50)

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # 保存损失曲线图
    plt.savefig(f'loss_curve_{ont}.png')
    plt.close()


    print("evaluating...")
    test_labels, test_probs = predict(model, test_loader)

    # 计算评估指标
    auc_score, aupr_score, f1_score, thresh = cal_metrics(test_probs, test_labels)

    print(f"测试集评估结果:")
    print(f"AUC: {auc_score:.4f}   AUPR: {aupr_score:.4f}   F1 Score: {f1_score:.4f}   thresh: {thresh:.4f}")

    # 保存模型
    torch.save(model.state_dict(), f'struct2go_{ont}.pth')
    print(f"模型已保存为 'struct2go_{ont}.pth'")