import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from imblearn.over_sampling import SMOTE

# 为了可复现性，设置种子
seed = 1
#设置Python标准库中模块的随机数生成器的种子。
random.seed(seed)
#设置Python标准库中模块的随机数生成器的种子。
np.random.seed(seed)
#设置TensorFlow库中的随机数生成器的种子，用于控制TensorFlow中涉及的随机操作
tf.random.set_seed(seed)

# 文件路径
train_path = r"C:\数字图像处理\图像分析\train.csv"
test_path = r"C:\数字图像处理\图像分析\testA.csv"

# 加载数据
train_df = pd.read_csv(train_path, index_col=0)
test_df = pd.read_csv(test_path, index_col=0)

# 特征工程
train_df = pd.concat([train_df, train_df['heartbeat_signals'].str.split(',', expand=True).astype(float)], axis=1)
#这一步将原始训练集的 DataFrame 与新生成的包含心跳信号数值的 DataFrame 进行列方向上的连接，即将新生成的列添加到原始 DataFrame 中。
test_df = pd.concat([test_df, test_df['heartbeat_signals'].str.split(',', expand=True).astype(float)], axis=1)

train_df.drop('heartbeat_signals', axis=1, inplace=True)
test_df.drop('heartbeat_signals', axis=1, inplace=True)

# 处理缺失值
train_df.fillna(train_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# 特征选择
X = train_df.drop("label", axis=1).values
y = train_df["label"].values
X_test = test_df.values

# 处理类别不平衡
sampler = SMOTE(random_state=seed, n_jobs=-1)
X, y = sampler.fit_resample(X, y)

# 特征缩放
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)



# Train-test split with stratified sampling
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

# 构建深度神经网络，并加入 dropout 层
model = Sequential()
#创建一个顺序（Sequential）模型
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
#添加一个Dropout层，以减少过拟合。
model.add(Dropout(0.5))
#添加第二个全连接层，包含128个神经元，同样使用ReLU作为激活函数。
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
#添加第二个全连接层，包含128个神经元，同样使用ReLU作为激活函数。
model.add(Dense(4, activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 学习率调度器
def lr_scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.9
    return lr

lr_schedule = LearningRateScheduler(lr_scheduler)

# 提前停止以防止过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 模型检查点以保存最佳模型
model_checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)

# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                    callbacks=[early_stopping, lr_schedule, model_checkpoint])


# 使用训练好的模型对验证集  进行预测，得到模型对每个样本的类别概率分布
y_pred = np.argmax(model.predict(X_val), axis=1)
print("Validation Classification Report:")
print(classification_report(y_val, y_pred))
print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# 创建一个新的图形对象
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
#创建一个新的图形对象
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#创建一个新的图形对象
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 保存预测结果
pred_probs = model.predict(X_test)
pred_df = pd.DataFrame(pred_probs, index=test_df.index, columns=[f"label_{i}" for i in range(4)])
pred_df.to_csv("submission4.csv")
pred_df.head()
