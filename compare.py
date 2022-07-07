import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
tf.config.experimental.set_virtual_device_configuration(gpus[0],
[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8096)])

def fasta_to_onehot(filename):

    with open(filename) as f:
        text = f.readlines()
    df_1 = pd.DataFrame([i.strip() for i in filter(lambda x: x[0]!=">",text)])
    set("".join(df_1[0])) # 不同的氨基酸
    AAs = ['Q', 'L', 'N', 'G', 'R', 'F', 'W', 'T', 'E', 'K', 'I', 'D', 'V', 'Y', 'S', 'A', 'C', 'M', 'H', 'P']
    X_train_Neg = df_1[0].apply(lambda x: pd.Series(list(x))).replace(AAs+["_"],range(21)).to_numpy()
    X_train_Neg = tf.one_hot(X_train_Neg,21)

    return X_train_Neg

filenames = ["DeepRMethylSite/test_s33_Neg_51.fasta","DeepRMethylSite/test_s33_Pos_51.fasta","DeepRMethylSite/train_s33_Neg_51.fasta","DeepRMethylSite/train_s33_Pos_51.fasta"]

X_test_Neg,X_test_Pos,X_train_Neg,X_train_Pos = [fasta_to_onehot(filename) for filename in filenames]

y_test_Neg,y_train_Neg = [tf.constant(np.array([0]*i)) for i in [x.shape[0] for x in [X_test_Neg,X_train_Neg]]]
y_test_Pos,y_train_Pos = [tf.constant(np.array([1]*i)) for i in [x.shape[0] for x in [X_test_Pos,X_train_Pos]]]

X_test = tf.concat([X_test_Pos,X_test_Neg],0)
X_train = tf.concat([X_train_Pos,X_train_Neg],0)

y_test = tf.concat([y_test_Pos,y_test_Neg],0)
y_train = tf.concat([y_train_Pos,y_train_Neg],0)


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


tf.random.set_seed(2022)
index = tf.random.shuffle(tf.constant(np.arange(len(X_train))))

# In [105]: index
# Out[105]: <tf.Tensor: shape=(252944,), dtype=int32, numpy=array([191521, 119163, 173193, ..., 160190,  65343, 237656])>

# 用验证集，选择超参数
tf.keras.backend.clear_session()
model = Model()
history = model.fit(tf.gather(X_train,index),tf.gather(y_train,index),512,100,2,validation_split=0.1)


y_true = y_test
y_pred = model.predict(X_test)[:,0]
auc_test = tf.metrics.AUC(1000)(y_true,y_pred)
print(auc_test)

# 保存模型权重
model.save_weights("model/model1_weights.hdf5")

# 保存和读取history.history
with open("model/model1_history.pkl","wb") as f:
    pickle.dump(history.history,f)
with open("model/model1_history.pkl","rb") as f:
    a = pickle.load(f)


# 超参数已经确定，不再使用验证集
tf.keras.backend.clear_session()
model = Model()
history = model.fit(tf.gather(X_train,index),tf.gather(y_train,index),512,100,2)

y_true = y_test
y_pred = model.predict(X_test)[:,0]
auc_test = tf.metrics.AUC(1000)(y_true,y_pred)
print(auc_test)

# 保存模型权重
model.save_weights("model/model2_weights.hdf5")

# 保存和读取history.history
with open("model/model2_history.pkl","wb") as f:
    pickle.dump(history.history,f)
with open("model/model2_history.pkl","rb") as f:
    a = pickle.load(f)


