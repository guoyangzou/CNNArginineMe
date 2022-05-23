import torch 
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
import random
import requests 
import math
#from pandarallel import pandarallel
from scipy import stats
#from bs4 import BeautifulSoup

## 分类任务
import sklearn.neighbors 
from sklearn.neighbors import KNeighborsClassifier  # K近邻
import sklearn.svm
from sklearn.svm import SVC  # 支持向量机
from sklearn.svm import OneClassSVM  # 
import sklearn.naive_bayes
from sklearn.naive_bayes import GaussianNB  # 朴素贝叶斯
import sklearn.tree  
from sklearn.tree import DecisionTreeClassifier  # 决策树
import sklearn.ensemble  
from sklearn.ensemble import BaggingClassifier  # 装袋法
from sklearn.ensemble import RandomForestClassifier  # 随机森林
import sklearn.neural_network
from sklearn.neural_network import MLPClassifier  # 神经网络
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.preprocessing import PolynomialFeatures  # 多项式回归
import sklearn.metrics
from sklearn.metrics import classification_report,roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_predict

from sklearn.manifold import TSNE,Isomap,MDS

from sklearn.decomposition import PCA  # 主成分分析
from sklearn.decomposition import TruncatedSVD  # 截断SVD和LSA
from sklearn.decomposition import SparseCoder # 字典学习
from sklearn.decomposition import FactorAnalysis # 因子分析
from sklearn.decomposition import FastICA  # 独立成分分析  
from sklearn.decomposition import NMF  # 非负矩阵分解
from sklearn.decomposition import LatentDirichletAllocation  # LDA

#gpus = tf.config.experimental.list_physical_devices(device_type="GPU")

#tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6096)])

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.compile(optimizer=tf.optimizers.Adam()
                     ,loss=tf.losses.BinaryCrossentropy()
                     ,metrics=[tf.metrics.AUC(1000)]
                    )
        
        self.cnn1 = tf.keras.Sequential([
            tf.keras.layers.Reshape([51,21]),
            tf.keras.layers.Conv1D(256,9,1,"valid"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Dropout(0.7),
        ])
        self.cnn2 = tf.keras.Sequential([
            tf.keras.layers.Reshape([21,256]),
            tf.keras.layers.Conv1D(32,7,1,"valid"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Dropout(0.5),
        ])
        self.simple = tf.keras.Sequential([
            tf.keras.layers.Reshape([7*32]),
            tf.keras.layers.Dense(128),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1,activation="sigmoid"),
        ])
    def call(self, inputs):
        
        x = self.cnn1(inputs)
        x = self.cnn2(x)
        x = self.simple(x)
        
        return x
    
model = Model()
model.build((None,51*21))
model.summary()

# datasets construction
df_1 = pd.read_csv("inputs/Methylation_site_dataset",sep='\t',header=2)

#df_2 = pd.read_csv("Methylation.txt",sep='\t',header=None)

df_1.query("ORGANISM=='human'",inplace=True)

df_1 = df_1[df_1.MOD_RSD.str[0]=="R"].copy()

df_1["Position"] = df_1.MOD_RSD.str.split("-").str[0].str[1:].astype(int)

df_1["Type"] = df_1.MOD_RSD.str.split("-").str[1]

df_1["SeqWin"] = df_1["SITE_+/-7_AA"].str.upper()

df_1 = df_1[["ACC_ID","Position","Type","SeqWin"]].copy()

df_1.Type.value_counts()

with open('inputs/uniprot_sp_isoform.fasta') as f:
    text = f.read()

p1 = re.compile("^>", re.M)

p2 = re.compile("^>\w\w\|(.+?)\|", re.M)

Entry = p2.findall(text)

Sequence = ["".join(i.split("\n")[1:]) for i in text.split("\n>")]

df_uniprot = pd.DataFrame(dict(Entry=Entry,Sequence=Sequence))

df_uniprot

df_1 = df_1.merge(df_uniprot,left_on="ACC_ID",right_on="Entry")

df_1 = df_1[["Entry","Position","Type","SeqWin","Sequence"]].copy()

def F_pad(seq,n=30):
    return "_"*n+seq+"_"*n

df_1.Sequence.apply(F_pad)

df_1.SeqWin.str.len().unique()

df_1.apply(lambda x:F_pad(x.Sequence,7).find(x.SeqWin),1)

df_1 = df_1[df_1.apply(lambda x:F_pad(x.Sequence,7).find(x.SeqWin)==x.Position-1,1)].copy()

df_1.SeqWin = df_1.apply(lambda x:F_pad(x.Sequence,25)[x.Position-1:x.Position+25*2],1)

set().issubset(set())

AAs = ['Q', 'L', 'N', 'G', 'R', 'F', 'W', 'T', 'E', 'K', 'I', 'D', 'V', 'Y', 'S', 'A', 'C', 'M', 'H', 'P']

df_1 = df_1[df_1.Sequence.apply(lambda x: set(x).issubset(AAs))].copy()

Pos_SeqWin_R = df_1.SeqWin

SeqWin_R = list()
for seq in df_1.Sequence:
    seq = F_pad(seq)
    for i,AA in enumerate(seq):
        if AA=="R":
            SeqWin_R.append(seq[i-25:i+1+25])



All_SeqWin_R = pd.Series(SeqWin_R).drop_duplicates()

Neg_SeqWin_R = All_SeqWin_R[~All_SeqWin_R.isin(Pos_SeqWin_R)]

Neg_SeqWin_R.to_csv("./temp/Neg_SeqWin_R")

Neg_SeqWin_R = pd.read_csv("./temp/Neg_SeqWin_R")["0"]

Neg_SeqWin_R

df_data_neg = Neg_SeqWin_R.apply(lambda x: pd.Series(list(x))).replace(AAs+["_"],range(21))

df_data_neg["label"] = 0

df_data_neg = df_data_neg.sample(80000,random_state=2021)

df_1_m1 = df_1.query("Type=='m1'")

df_data_m1 = df_1_m1.SeqWin.drop_duplicates().apply(lambda x: pd.Series(list(x))).replace(AAs+["_"],range(21))

df_data_m1["label"] = 1

df_data_m1


df_1_m2 = df_1.query("Type=='m2'")

df_data_m2 = df_1_m2.SeqWin.drop_duplicates().apply(lambda x: pd.Series(list(x))).replace(AAs+["_"],range(21))

df_data_m2["label"] = 1

df_data_m2


df_data_train_m1 = pd.concat([df_data_m1,df_data_neg[:40000]])

df_data_train_m2 = pd.concat([df_data_m2,df_data_neg[40000:]])

data_train_m1 = df_data_train_m1.to_numpy()

data_train_m2 = df_data_train_m2.to_numpy()

data_train_me = np.concatenate([data_train_m1,data_train_m2])

data_train_m1
data_train_m1
np.random.shuffle(data_train_m1)

x_train_m1 = tf.one_hot(data_train_m1[:len(data_train_m1)//10*9][:,:51],21).numpy().reshape([-1,51*21])

y_train_m1 = data_train_m1[:len(data_train_m1)//10*9][:,-1:]

x_test_m1 = tf.one_hot(data_train_m1[len(data_train_m1)//10*9:][:,:51],21).numpy().reshape([-1,51*21])

y_test_m1 = data_train_m1[len(data_train_m1)//10*9:][:,-1:]

data_train_m2
np.random.shuffle(data_train_m2)

x_train_m2 = tf.one_hot(data_train_m2[:len(data_train_m2)//10*9][:,:51],21).numpy().reshape([-1,51*21])

y_train_m2 = data_train_m2[:len(data_train_m2)//10*9][:,-1:]

x_test_m2 = tf.one_hot(data_train_m2[len(data_train_m2)//10*9:][:,:51],21).numpy().reshape([-1,51*21])

y_test_m2 = data_train_m2[len(data_train_m2)//10*9:][:,-1:]

data_train_me
np.random.shuffle(data_train_me)

x_train_me = tf.one_hot(data_train_me[:len(data_train_me)//10*9][:,:51],21).numpy().reshape([-1,51*21])

y_train_me = data_train_me[:len(data_train_me)//10*9][:,-1:]

x_test_me = tf.one_hot(data_train_me[len(data_train_me)//10*9:][:,:51],21).numpy().reshape([-1,51*21])

y_test_me = data_train_me[len(data_train_me)//10*9:][:,-1:]

torch.save([x_train_m1,y_train_m1,x_train_m2,y_train_m2,x_train_me,y_train_me],"./temp/x_train_m1")
torch.save([x_test_m1,y_test_m1,x_test_m2,y_test_m2,x_test_me,y_test_me],"./temp/x_test_m1")

# load datasets of test and training
x_train_m1,y_train_m1,x_train_m2,y_train_m2,x_train_me,y_train_me = torch.load("./temp/x_train_m1")
x_test_m1,y_test_m1,x_test_m2,y_test_m2,x_test_me,y_test_me = torch.load("./temp/x_test_m1")

x_train_m1_list = list()
x_valid_m1_list = list()

Len = x_train_m1.shape[0]//10

for i in range(10):
    x_valid_m1_list.append(x_train_m1[Len*i:Len*(i+1)])
    x_train_m1_list.append(x_train_m1[np.r_[0:Len*i,Len*(i+1):Len*10]])
    
y_train_m1_list = list()
y_valid_m1_list = list()

Len = y_train_m1.shape[0]//10

for i in range(10):
    y_valid_m1_list.append(y_train_m1[Len*i:Len*(i+1)])
    y_train_m1_list.append(y_train_m1[np.r_[0:Len*i,Len*(i+1):Len*10]])

# drop redundancy for test datasets
temp1 = pd.DataFrame(x_train_m1[y_train_m1[:,0]==1].reshape([-1,51,21]).argmax(2)).replace(np.arange(21),AAs+["_"])

temp2 = temp1.apply(lambda x: "".join(x),1)

with open("temp/x_train_m1_peptide_pos.txt","w") as f:
    s = "\n".join(">"+temp2.index.to_series().apply(str)+"\n"+temp2)
    #f.write(s)

temp1 = pd.DataFrame(x_test_m1[y_test_m1[:,0]==1].reshape([-1,51,21]).argmax(2)).replace(np.arange(21),AAs+["_"])

temp2 = temp1.apply(lambda x: "".join(x),1)

with open("temp/x_test_m1_peptide_pos.txt","w") as f:
    s = "\n".join(">"+temp2.index.to_series().apply(str)+"\n"+temp2)
    #f.write(s)
with open("inputs/1618532184.fas.db2novel.clstr.sorted") as f:
    cluster = f.read()

redundancy = [int(k) for k in "_".join(["_".join([j.split(">")[1].split("...")[0] for j in i.split("\n")[2:-1]]) for i in cluster.split(">Cluster")[1:]][:3014]).split("_")]

nonredundancy = pd.Series(np.arange(len(x_test_m1)))[~pd.Series(np.arange(len(x_test_m1))).isin(redundancy)].to_numpy()


# training
import os,sys

[os.makedirs("outputs/DL_Model/weights/%02d"%i,exist_ok=True) for i in range(1,11)]

ModelCheckpoints = list()
for i in range(10):
    ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath="outputs/DL_Model/weights/%02d/{epoch:03d}-{val_auc:.3f}.hdf5"%(i+1),monitor="val_auc",save_weights_only=True,)
    ModelCheckpoints.append(ModelCheckpoint)

for i in range(10):
    tf.keras.backend.clear_session()
    tf.random.set_seed(2021)
    model = Model()
    history = model.fit(x_train_m1_list[i],y_train_m1_list[i],512,100,2
              ,validation_data=(x_valid_m1_list[i],y_valid_m1_list[i])
              ,class_weight={1:1,0:1/30}
              ,callbacks=[ModelCheckpoints[i]]
             )

# performance test
auc_test_list = list()
SnSp90_test_list = list()
SnSp95_test_list = list()
SnSp99_test_list = list()

auc_test_list_non = list()
SnSp90_test_list_non = list()
SnSp95_test_list_non = list()
SnSp99_test_list_non = list()

auc_valid_list = list()
SnSp90_valid_list = list()
SnSp95_valid_list = list()
SnSp99_valid_list = list()
for i in range(10):
    model = Model()
    model.build((None,51*21))
    tf.keras.backend.clear_session()
    Se_1 = pd.Series(os.listdir("outputs/DL_Model/weights/%02d"%(i+1)))
    path = "outputs/DL_Model/weights/%02d/"%(i+1)+Se_1[Se_1.str.split("-").str[0]=="100"].values[0]
    
    model.load_weights(path)

    y_pred = model.predict(x_test_m1)[:,0]
    y_true = y_test_m1[:,0]
    auc_test = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_test = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_test = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_test = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_test_list.append(auc_test)
    SnSp90_test_list.append(SnSp90_test)
    SnSp95_test_list.append(SnSp95_test)
    SnSp99_test_list.append(SnSp99_test)
    
    y_pred = model.predict(x_test_m1[nonredundancy])[:,0]
    y_true = y_test_m1[nonredundancy][:,0]
    auc_test = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_test = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_test = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_test = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_test_list_non.append(auc_test)
    SnSp90_test_list_non.append(SnSp90_test)
    SnSp95_test_list_non.append(SnSp95_test)
    SnSp99_test_list_non.append(SnSp99_test)
    
    y_pred = model.predict(x_valid_m1_list[i])[:,0]
    y_true = y_valid_m1_list[i][:,0]
    auc_valid = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_valid = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_valid = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_valid = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_valid_list.append(auc_valid)
    SnSp90_valid_list.append(SnSp90_valid)
    SnSp95_valid_list.append(SnSp95_valid)
    SnSp99_valid_list.append(SnSp99_valid)
    
print("test:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")    
for i in range(10):
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_test_list[i],SnSp90_test_list[i],SnSp95_test_list[i],SnSp99_test_list[i]
    ))
    
print("test_nonredundancy:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")    
for i in range(10):
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_test_list_non[i],SnSp90_test_list_non[i],SnSp95_test_list_non[i],SnSp99_test_list_non[i]
    ))

print("valid:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")  
for i in range(10):      
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_valid_list[i],SnSp90_valid_list[i],SnSp95_valid_list[i],SnSp99_valid_list[i]
    ))

# traditional machine learning model

## RandomForestClassifier
auc_test_list = list()
SnSp90_test_list = list()
SnSp95_test_list = list()
SnSp99_test_list = list()

auc_valid_list = list()
SnSp90_valid_list = list()
SnSp95_valid_list = list()
SnSp99_valid_list = list()
for i in range(10):
    model = RandomForestClassifier()

    model.fit(x_train_m1_list[i],y_train_m1_list[i].ravel())

    torch.save(model,f"outputs/Model/m1_RF_{i+1}.hdf5")

    model = torch.load(f"outputs/Model/m1_RF_{i+1}.hdf5")

    y_pred = model.predict_proba(x_test_m1)[:,1]
    y_true = y_test_m1[:,0]
    auc_test = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_test = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_test = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_test = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_test_list.append(auc_test)
    SnSp90_test_list.append(SnSp90_test)
    SnSp95_test_list.append(SnSp95_test)
    SnSp99_test_list.append(SnSp99_test)
    
    y_pred = model.predict_proba(x_valid_m1_list[i])[:,1]
    y_true = y_valid_m1_list[i][:,0]
    auc_valid = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_valid = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_valid = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_valid = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_valid_list.append(auc_valid)
    SnSp90_valid_list.append(SnSp90_valid)
    SnSp95_valid_list.append(SnSp95_valid)
    SnSp99_valid_list.append(SnSp99_valid)
    
print("test:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")    
for i in range(10):
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_test_list[i],SnSp90_test_list[i],SnSp95_test_list[i],SnSp99_test_list[i]
    ))

print("valid:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")  
for i in range(10):      
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_valid_list[i],SnSp90_valid_list[i],SnSp95_valid_list[i],SnSp99_valid_list[i]
    ))

print("test:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")
print(np.mean(auc_test_list),np.mean(SnSp90_test_list),np.mean(SnSp95_test_list),np.mean(SnSp99_test_list))  
print("valid:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")  
print(np.mean(auc_valid_list),np.mean(SnSp90_valid_list),np.mean(SnSp95_valid_list),np.mean(SnSp99_valid_list))

## SVC
auc_test_list = list()
SnSp90_test_list = list()
SnSp95_test_list = list()
SnSp99_test_list = list()

auc_valid_list = list()
SnSp90_valid_list = list()
SnSp95_valid_list = list()
SnSp99_valid_list = list()
for i in range(10):
    model = SVC(probability=True)

    model.fit(x_train_m1_list[i],y_train_m1_list[i].ravel())

    torch.save(model,f"outputs/Model/m1_SVC_{i+1}.hdf5")

    model = torch.load(f"outputs/Model/m1_SVC_{i+1}.hdf5")

    y_pred = model.predict_proba(x_test_m1)[:,1]
    y_true = y_test_m1[:,0]
    auc_test = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_test = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_test = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_test = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_test_list.append(auc_test)
    SnSp90_test_list.append(SnSp90_test)
    SnSp95_test_list.append(SnSp95_test)
    SnSp99_test_list.append(SnSp99_test)
    
    y_pred = model.predict_proba(x_valid_m1_list[i])[:,1]
    y_true = y_valid_m1_list[i][:,0]
    auc_valid = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_valid = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_valid = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_valid = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_valid_list.append(auc_valid)
    SnSp90_valid_list.append(SnSp90_valid)
    SnSp95_valid_list.append(SnSp95_valid)
    SnSp99_valid_list.append(SnSp99_valid)

print("test:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")    
for i in range(10):
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_test_list[i],SnSp90_test_list[i],SnSp95_test_list[i],SnSp99_test_list[i]
    ))

print("valid:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")  
for i in range(10):      
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_valid_list[i],SnSp90_valid_list[i],SnSp95_valid_list[i],SnSp99_valid_list[i]
    ))

print("test:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")
print(np.mean(auc_test_list),np.mean(SnSp90_test_list),np.mean(SnSp95_test_list),np.mean(SnSp99_test_list))  
print("valid:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")  
print(np.mean(auc_valid_list),np.mean(SnSp90_valid_list),np.mean(SnSp95_valid_list),np.mean(SnSp99_valid_list))

## KNeighborsClassifier
auc_test_list = list()
SnSp90_test_list = list()
SnSp95_test_list = list()
SnSp99_test_list = list()

auc_valid_list = list()
SnSp90_valid_list = list()
SnSp95_valid_list = list()
SnSp99_valid_list = list()
for i in range(10):
    model = KNeighborsClassifier()

    model.fit(x_train_m1_list[i],y_train_m1_list[i].ravel())

    torch.save(model,f"outputs/Model/m1_KN_{i+1}.hdf5")

    model = torch.load(f"outputs/Model/m1_KN_{i+1}.hdf5")

    y_pred = model.predict_proba(x_test_m1)[:,1]
    y_true = y_test_m1[:,0]
    auc_test = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_test = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_test = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_test = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_test_list.append(auc_test)
    SnSp90_test_list.append(SnSp90_test)
    SnSp95_test_list.append(SnSp95_test)
    SnSp99_test_list.append(SnSp99_test)
    
    y_pred = model.predict_proba(x_valid_m1_list[i])[:,1]
    y_true = y_valid_m1_list[i][:,0]
    auc_valid = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_valid = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_valid = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_valid = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_valid_list.append(auc_valid)
    SnSp90_valid_list.append(SnSp90_valid)
    SnSp95_valid_list.append(SnSp95_valid)
    SnSp99_valid_list.append(SnSp99_valid)
    
print("test:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")    
for i in range(10):
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_test_list[i],SnSp90_test_list[i],SnSp95_test_list[i],SnSp99_test_list[i]
    ))

print("valid:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")  
for i in range(10):      
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_valid_list[i],SnSp90_valid_list[i],SnSp95_valid_list[i],SnSp99_valid_list[i]
    ))

print("test:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")
print(np.mean(auc_test_list),np.mean(SnSp90_test_list),np.mean(SnSp95_test_list),np.mean(SnSp99_test_list))  
print("valid:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")  
print(np.mean(auc_valid_list),np.mean(SnSp90_valid_list),np.mean(SnSp95_valid_list),np.mean(SnSp99_valid_list))

## GaussianNB

auc_test_list = list()
SnSp90_test_list = list()
SnSp95_test_list = list()
SnSp99_test_list = list()

auc_valid_list = list()
SnSp90_valid_list = list()
SnSp95_valid_list = list()
SnSp99_valid_list = list()
for i in range(10):
    model = GaussianNB()

    model.fit(x_train_m1_list[i],y_train_m1_list[i].ravel())

    torch.save(model,f"outputs/Model/m1_GN_{i+1}.hdf5")

    model = torch.load(f"outputs/Model/m1_GN_{i+1}.hdf5")

    y_pred = model.predict_proba(x_test_m1)[:,1]
    y_true = y_test_m1[:,0]
    auc_test = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_test = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_test = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_test = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_test_list.append(auc_test)
    SnSp90_test_list.append(SnSp90_test)
    SnSp95_test_list.append(SnSp95_test)
    SnSp99_test_list.append(SnSp99_test)
    
    y_pred = model.predict_proba(x_valid_m1_list[i])[:,1]
    y_true = y_valid_m1_list[i][:,0]
    auc_valid = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_valid = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_valid = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_valid = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_valid_list.append(auc_valid)
    SnSp90_valid_list.append(SnSp90_valid)
    SnSp95_valid_list.append(SnSp95_valid)
    SnSp99_valid_list.append(SnSp99_valid)
    
print("test:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")    
for i in range(10):
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_test_list[i],SnSp90_test_list[i],SnSp95_test_list[i],SnSp99_test_list[i]
    ))

print("valid:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")  
for i in range(10):      
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_valid_list[i],SnSp90_valid_list[i],SnSp95_valid_list[i],SnSp99_valid_list[i]
    ))

print("test:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")
print(np.mean(auc_test_list),np.mean(SnSp90_test_list),np.mean(SnSp95_test_list),np.mean(SnSp99_test_list))  
print("valid:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")  
print(np.mean(auc_valid_list),np.mean(SnSp90_valid_list),np.mean(SnSp95_valid_list),np.mean(SnSp99_valid_list))

## DecisionTreeClassifier
auc_test_list = list()
SnSp90_test_list = list()
SnSp95_test_list = list()
SnSp99_test_list = list()

auc_valid_list = list()
SnSp90_valid_list = list()
SnSp95_valid_list = list()
SnSp99_valid_list = list()
for i in range(10):
    model = DecisionTreeClassifier()

    model.fit(x_train_m1_list[i],y_train_m1_list[i].ravel())

    torch.save(model,f"outputs/Model/m1_DT_{i+1}.hdf5")

    model = torch.load(f"outputs/Model/m1_DT_{i+1}.hdf5")

    y_pred = model.predict_proba(x_test_m1)[:,1]
    y_true = y_test_m1[:,0]
    auc_test = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_test = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_test = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_test = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_test_list.append(auc_test)
    SnSp90_test_list.append(SnSp90_test)
    SnSp95_test_list.append(SnSp95_test)
    SnSp99_test_list.append(SnSp99_test)
    
    y_pred = model.predict_proba(x_valid_m1_list[i])[:,1]
    y_true = y_valid_m1_list[i][:,0]
    auc_valid = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_valid = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_valid = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_valid = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_valid_list.append(auc_valid)
    SnSp90_valid_list.append(SnSp90_valid)
    SnSp95_valid_list.append(SnSp95_valid)
    SnSp99_valid_list.append(SnSp99_valid)
    
print("test:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")    
for i in range(10):
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_test_list[i],SnSp90_test_list[i],SnSp95_test_list[i],SnSp99_test_list[i]
    ))

print("valid:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")  
for i in range(10):      
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_valid_list[i],SnSp90_valid_list[i],SnSp95_valid_list[i],SnSp99_valid_list[i]
    ))

print("test:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")
print(np.mean(auc_test_list),np.mean(SnSp90_test_list),np.mean(SnSp95_test_list),np.mean(SnSp99_test_list))  
print("valid:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")  
print(np.mean(auc_valid_list),np.mean(SnSp90_valid_list),np.mean(SnSp95_valid_list),np.mean(SnSp99_valid_list))

## BaggingClassifier
auc_test_list = list()
SnSp90_test_list = list()
SnSp95_test_list = list()
SnSp99_test_list = list()

auc_valid_list = list()
SnSp90_valid_list = list()
SnSp95_valid_list = list()
SnSp99_valid_list = list()
for i in range(10):
    model = BaggingClassifier()

    model.fit(x_train_m1_list[i],y_train_m1_list[i].ravel())

    torch.save(model,f"outputs/Model/m1_BC_{i+1}.hdf5")

    model = torch.load(f"outputs/Model/m1_BC_{i+1}.hdf5")

    y_pred = model.predict_proba(x_test_m1)[:,1]
    y_true = y_test_m1[:,0]
    auc_test = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_test = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_test = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_test = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_test_list.append(auc_test)
    SnSp90_test_list.append(SnSp90_test)
    SnSp95_test_list.append(SnSp95_test)
    SnSp99_test_list.append(SnSp99_test)
    
    y_pred = model.predict_proba(x_valid_m1_list[i])[:,1]
    y_true = y_valid_m1_list[i][:,0]
    auc_valid = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_valid = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_valid = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_valid = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_valid_list.append(auc_valid)
    SnSp90_valid_list.append(SnSp90_valid)
    SnSp95_valid_list.append(SnSp95_valid)
    SnSp99_valid_list.append(SnSp99_valid)
    
print("test:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")    
for i in range(10):
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_test_list[i],SnSp90_test_list[i],SnSp95_test_list[i],SnSp99_test_list[i]
    ))

print("valid:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")  
for i in range(10):      
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_valid_list[i],SnSp90_valid_list[i],SnSp95_valid_list[i],SnSp99_valid_list[i]
    ))

print("test:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")
print(np.mean(auc_test_list),np.mean(SnSp90_test_list),np.mean(SnSp95_test_list),np.mean(SnSp99_test_list))  
print("valid:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")  
print(np.mean(auc_valid_list),np.mean(SnSp90_valid_list),np.mean(SnSp95_valid_list),np.mean(SnSp99_valid_list))

## LogisticRegression
auc_test_list = list()
SnSp90_test_list = list()
SnSp95_test_list = list()
SnSp99_test_list = list()

auc_valid_list = list()
SnSp90_valid_list = list()
SnSp95_valid_list = list()
SnSp99_valid_list = list()
for i in range(10):
    model = LogisticRegression(max_iter=1000)

    model.fit(x_train_m1_list[i],y_train_m1_list[i].ravel())

    torch.save(model,f"outputs/Model/m1_LR_{i+1}.hdf5")

    model = torch.load(f"outputs/Model/m1_LR_{i+1}.hdf5")

    y_pred = model.predict_proba(x_test_m1)[:,1]
    y_true = y_test_m1[:,0]
    auc_test = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_test = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_test = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_test = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_test_list.append(auc_test)
    SnSp90_test_list.append(SnSp90_test)
    SnSp95_test_list.append(SnSp95_test)
    SnSp99_test_list.append(SnSp99_test)
    
    y_pred = model.predict_proba(x_valid_m1_list[i])[:,1]
    y_true = y_valid_m1_list[i][:,0]
    auc_valid = tf.metrics.AUC(1000)(y_true,y_pred)
    SnSp90_valid = tf.metrics.SensitivityAtSpecificity(0.9)(y_true,y_pred)
    SnSp95_valid = tf.metrics.SensitivityAtSpecificity(0.95)(y_true,y_pred)
    SnSp99_valid = tf.metrics.SensitivityAtSpecificity(0.99)(y_true,y_pred)
    auc_valid_list.append(auc_valid)
    SnSp90_valid_list.append(SnSp90_valid)
    SnSp95_valid_list.append(SnSp95_valid)
    SnSp99_valid_list.append(SnSp99_valid)
    
print("test:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")    
for i in range(10):
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_test_list[i],SnSp90_test_list[i],SnSp95_test_list[i],SnSp99_test_list[i]
    ))

print("valid:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")  
for i in range(10):      
    print("%5.3f %5.3f %5.3f %5.3f"%(
        auc_valid_list[i],SnSp90_valid_list[i],SnSp95_valid_list[i],SnSp99_valid_list[i]
    ))

print("test:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")
print(np.mean(auc_test_list),np.mean(SnSp90_test_list),np.mean(SnSp95_test_list),np.mean(SnSp99_test_list))  
print("valid:AUC, Sn(Sp=0.9), Sn(Sp=0.95), Sn(Sp=0.99)")  
print(np.mean(auc_valid_list),np.mean(SnSp90_valid_list),np.mean(SnSp95_valid_list),np.mean(SnSp99_valid_list))

## plot ROC 
plt.figure(figsize=[10,7])
#plt.plot(fpr,tpr,"C1",lw=2,label="AUC_DL=%.3f"%(0.847))

model = torch.load("outputs/Model/m1_RF_1.hdf5")
y_pred = model.predict_proba(x_test_m1[nonredundancy])[:,1]
roc_auc_score(y_true,y_pred)
fpr1,tpr1,thresholds1 = roc_curve(y_true,y_pred)
plt.plot(fpr1,tpr1,"C2",lw=2,label="AUC_RF=%.3f"%roc_auc_score(y_true,y_pred))

model = torch.load("outputs/Model/m1_GN_1.hdf5")
y_pred = model.predict_proba(x_test_m1[nonredundancy])[:,1]
roc_auc_score(y_true,y_pred)
fpr1,tpr1,thresholds1 = roc_curve(y_true,y_pred)
plt.plot(fpr1,tpr1,"C3",lw=2,label="AUC_GN=%.3f"%roc_auc_score(y_true,y_pred))

model = torch.load("outputs/Model/m1_DT_1.hdf5")
y_pred = model.predict_proba(x_test_m1[nonredundancy])[:,1]
roc_auc_score(y_true,y_pred)
fpr1,tpr1,thresholds1 = roc_curve(y_true,y_pred)
plt.plot(fpr1,tpr1,"C4",lw=2,label="AUC_DT=%.3f"%roc_auc_score(y_true,y_pred))

model = torch.load("outputs/Model/m1_BC_1.hdf5")
y_pred = model.predict_proba(x_test_m1[nonredundancy])[:,1]
roc_auc_score(y_true,y_pred)
fpr1,tpr1,thresholds1 = roc_curve(y_true,y_pred)
plt.plot(fpr1,tpr1,"C5",lw=2,label="AUC_BC=%.3f"%roc_auc_score(y_true,y_pred))

model = torch.load("outputs/Model/m1_LR_1.hdf5")
y_pred = model.predict_proba(x_test_m1[nonredundancy])[:,1]
roc_auc_score(y_true,y_pred)
fpr1,tpr1,thresholds1 = roc_curve(y_true,y_pred)
plt.plot(fpr1,tpr1,"C6",lw=2,label="AUC_LR=%.3f"%roc_auc_score(y_true,y_pred))

model = torch.load("outputs/Model/m1_KN_1.hdf5")
y_pred = model.predict_proba(x_test_m1[nonredundancy])[:,1]
roc_auc_score(y_true,y_pred)
fpr1,tpr1,thresholds1 = roc_curve(y_true,y_pred)
plt.plot(fpr1,tpr1,"C7",lw=2,label="AUC_KN=%.3f"%roc_auc_score(y_true,y_pred))

model = torch.load("outputs/Model/m1_SVC_1.hdf5")
y_pred = model.predict_proba(x_test_m1[nonredundancy])[:,1]
roc_auc_score(y_true,y_pred)
fpr1,tpr1,thresholds1 = roc_curve(y_true,y_pred)
plt.plot(fpr1,tpr1,"C8",lw=2,label="AUC_SVC=%.3f"%roc_auc_score(y_true,y_pred))

plt.xlim(0,1)
plt.ylim(0,1)
plt.xticks([0,0.1,1])
plt.yticks([1])
plt.xlabel("1-Sp",fontsize=20)
plt.ylabel("Sn",fontsize=20)
plt.title("ROC",fontsize=20)
plt.legend(fontsize=15)
#plt.show()
plt.savefig("outputs/ROC_multiple.pdf")

C = ["C1","C2","C3","C4","C5","C6","C7","C8"]
M = ["DL","RF","GN","DT","BC","LR","KN","SVC"]
plt.figure(figsize=[15,25])
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=None,hspace=0.3)
for i in range(8):
    if i == 0:
        plt.subplot(4,2,i+1)
        plt.plot(fpr,tpr,C[i],lw=2,label="AUC_%s=%.3f"%(M[i],0.847))
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xticks([0,0.1,1])
        plt.yticks([1])
        plt.xlabel("1-Sp",fontsize=15)
        plt.ylabel("Sn",fontsize=15)
        plt.title("ROC_%s"%M[i],fontsize=15)
        plt.legend(fontsize=12)
    else:   
        plt.subplot(4,2,i+1)
        model = torch.load("outputs/Model/m1_%s_1.hdf5"%M[i])
        y_pred = model.predict_proba(x_test_m1[nonredundancy])[:,1]
        roc_auc_score(y_true,y_pred)
        fpr1,tpr1,thresholds1 = roc_curve(y_true,y_pred)
        plt.plot(fpr1,tpr1,C[i],lw=2,label="AUC_%s=%.3f"%(M[i],roc_auc_score(y_true,y_pred)))

        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xticks([0,0.1,1])
        plt.yticks([1])
        plt.xlabel("1-Sp",fontsize=15)
        plt.ylabel("Sn",fontsize=15)
        plt.title("ROC_%s"%M[i],fontsize=15)
        plt.legend(fontsize=12)
#plt.show()
plt.savefig("outputs/ROC_single.pdf")