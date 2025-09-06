2025.9.5
baseline: struct2go

加入标签传递，使用氨基酸独热编码，数据集参数8:1:1，损失函数BCELoss，epoch=50
模型参数SAGNetworkHierarchical(20, 512, n_labels, 2560, num_convs=3, pool_ratio=0.4, dropout=0.5)

bp: Total samples: 56852, Features: 2560, Labels: 25699
AUC: 0.9809   AUPR: 0.3970   F1 Score: 0.4149   thresh: 0.2009

cc: Total samples: 56871, Features: 2560, Labels: 4052
AUC: 0.9928   AUPR: 0.7242   F1 Score: 0.6618   thresh: 0.3105

mf: Total samples: 51572, Features: 2560, Labels: 10155
AUC: 0.9924   AUPR: 0.7217   F1 Score: 0.6851   thresh: 0.3097

all: Total samples: 59022, Features: 2560, Labels: 39906
AUC: 0.9846   AUPR: 0.5051   F1 Score: 0.4981   thresh: 0.2441