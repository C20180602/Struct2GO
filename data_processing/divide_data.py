import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataSet(Dataset):
    def __init__(self,emb_graph,emb_seq_feature,emb_label):
        super().__init__()
        self.list = list(emb_graph.keys())
        self.graphs = emb_graph
        self.seq_feature = emb_seq_feature
        self.label = emb_label

    def __getitem__(self,index): 
        protein = self.list[index] 
        graph = self.graphs[protein]
        seq_feature = self.seq_feature[protein]
        label = self.label[protein]

        return graph, label, seq_feature 

    def __len__(self):
        return  len(self.list) 

if __name__ == "__main__":
    ns_type_list = ['bp','mf','cc']
    for ns_type in ns_type_list:
        print("divide",ns_type,"dataset")
        with open('../processed_data/emb_graph_'+ns_type,'rb')as f:
            emb_graph = pickle.load(f)
        with open('../processed_data/emb_seq_feature_'+ns_type,'rb')as f:
            emb_seq_feature = pickle.load(f)
        with open('../processed_data/emb_label_'+ns_type,'rb')as f:
            emb_label = pickle.load(f)

        dataset = MyDataSet(emb_graph = emb_graph, emb_seq_feature = emb_seq_feature, emb_label = emb_label)
        train_size = int(len(dataset) * 0.7)
        valid_size = int(len(dataset) * 0.2)
        test_size = len(dataset) - train_size - valid_size
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
        with open('../divided_data/'+ns_type+'_train_dataset','wb')as f:
            pickle.dump(train_dataset,f)
        with open('../divided_data/'+ns_type+'_valid_dataset','wb')as f:
            pickle.dump(valid_dataset,f)
        with open('../divided_data/'+ns_type+'_test_dataset','wb')as f:
            pickle.dump(test_dataset,f)
        print("train dataset size",len(train_dataset))
        print("valid dataset size",len(valid_dataset))
        print("test dataset size",len(test_dataset))