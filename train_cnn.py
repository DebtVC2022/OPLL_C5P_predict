from cProfile import label
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from torch.optim.lr_scheduler import StepLR

# 假设你的特征数据保存在X中，标签保存在y中
excel_file = "OPLL-机器学习-C5p患者资料.xlsx"
sheet_name = "append_data_unbalance"
df = pd.read_excel(excel_file, sheet_name=sheet_name)
X = df.iloc[:, 1:]
X = X.fillna(-99)
y = df.iloc[:, 0]

X = np.array(X)# 你的特征数据 (240x32的矩阵)
y = np.array(y)# 你的标签数据 (240个元素的向量)

# 划分数据集为训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=44)

class ModifiedConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            # 卷积层计算
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            # 批归一化
            nn.BatchNorm1d(32),
            # ReLU激活函数
            nn.ReLU(),
            # 池化层：最大池化
            nn.MaxPool1d(kernel_size=2, stride=2))
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        
        # 计算全连接层的输入特征数量
        fc_input_size = self._get_fc_input_size(23, num_conv_layers=2)
        
        self.fc = nn.Linear(320, 1)

        self.sigmoid = nn.Sigmoid()
    
    def _get_fc_input_size(self, input_size, num_conv_layers):
        # 计算全连接层的输入特征数量
        for _ in range(num_conv_layers):
            input_size = (input_size - 5 + 2) // 1 + 1  # 计算卷积层的输出维度
            input_size = (input_size - 2) // 2 + 1  # 计算池化层的输出维度
        return input_size * 64  # 64 是第二个卷积层的输出通道数
    
    # 定义前向传播顺序
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(x.shape[0], -1)  # 将张量展平以传递给全连接层
        out = self.sigmoid(self.fc(out))
        return out
    

class ModifiedConvNet1(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedConvNet1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1)
        )
        self.fc = nn.Linear(352, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.conv1(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


model = ModifiedConvNet1()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 平衡因子，可用于调整类别权重
        self.gamma = gamma  # 调整焦点损失的聚焦度
        self.reduction = reduction  # 损失的缩减方式

    def forward(self, input, target):
        # 计算交叉熵损失
        cross_entropy = nn.functional.binary_cross_entropy_with_logits(input, target, reduction='none')

        # 计算焦点损失
        p = torch.exp(-cross_entropy)
        focal_loss = (self.alpha * (1 - p) ** self.gamma * cross_entropy)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失
#criterion = FocalLoss(alpha=1, gamma=2)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 初始化计数器
eval_counter = 0

# 定义批处理大小
batch_size = 16

# 将数据转换为TensorDataset
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练模型
loss_all_list = []
for epoch in range(1, 501):
    model.train()

    for inputs, labels in train_loader:
        # 正向传播
        labels = labels.reshape(-1, 1)
        outputs = model(inputs.unsqueeze(1))
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    if epoch % 2 == 0:
        loss_all_list.append(loss.detach().numpy())
        model.eval()
        val_total_outputs = []
        val_total_labels = []

        for val_inputs, val_labels in val_loader:
            val_outputs = model(val_inputs.unsqueeze(1))
            val_total_outputs.append(val_outputs)
            val_total_labels.append(val_labels)

        val_outputs_all = torch.cat(val_total_outputs)
        val_labels_all = torch.cat(val_total_labels)

        val_auc = roc_auc_score(val_labels_all.detach().numpy(), val_outputs_all.detach().numpy())

        val_outputs_all = (val_outputs_all > 0.5).float()
        val_accuracy = accuracy_score(val_labels_all.detach().numpy(), val_outputs_all.detach().numpy())
        val_precision = precision_score(val_labels_all.detach().numpy(), val_outputs_all.detach().numpy())
        val_recall = recall_score(val_labels_all.detach().numpy(), val_outputs_all.detach().numpy())
        val_f1 = f1_score(val_labels_all.detach().numpy(), val_outputs_all.detach().numpy())

        print(f"Evaluation {eval_counter} (Epoch {epoch}):")
        print(f"准确率: {val_accuracy}")
        print(f"精确度: {val_precision}")
        print(f"召回率: {val_recall}")
        print(f"F1分数: {val_f1}")
        print(f"AUC分数: {val_auc}")

# 在测试集上评估模型
model.eval()
test_total_outputs = []
test_total_labels = []

for test_inputs, test_labels in test_loader:
    test_outputs = model(test_inputs.unsqueeze(1))
    test_total_outputs.append(test_outputs)
    test_total_labels.append(test_labels)

test_outputs_all = torch.cat(test_total_outputs)
test_labels_all = torch.cat(test_total_labels)
print(test_labels_all)
test_auc = roc_auc_score(test_labels_all.detach().numpy(), test_outputs_all.detach().numpy())
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(test_labels_all.detach().numpy(), test_outputs_all.detach().numpy())

test_outputs_all = (test_outputs_all > 0.5).float()
print(test_outputs_all[0])
test_accuracy = accuracy_score(test_labels_all.detach().numpy(), test_outputs_all.detach().numpy())
test_precision = precision_score(test_labels_all.detach().numpy(), test_outputs_all.detach().numpy())
test_recall = recall_score(test_labels_all.detach().numpy(), test_outputs_all.detach().numpy())
test_f1 = f1_score(test_labels_all.detach().numpy(), test_outputs_all.detach().numpy())


print("在测试集上的性能评估:")
print(f"准确率: {test_accuracy}")
print(f"精确度: {test_precision}")
print(f"召回率: {test_recall}")
print(f"F1分数: {test_f1}")
print(f"AUC分数: {test_auc}")

import pandas as pd
pd.DataFrame(np.array(loss_all_list).reshape(-1, 1)).to_csv("1_cnn_celoss_loss_list.csv")

import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(10, 6))
plt.plot(pd.DataFrame(np.array(loss_all_list).reshape(-1, 1)), label="The loss function curve with 1-layer cnn model")
plt.xlabel("Epoch",weight='bold')
plt.ylabel("Loss",weight='bold')
plt.legend(prop={'size':16})
plt.savefig("1_cnn_celoss_loss_list_plot.png")




# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {test_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)',weight='bold')
plt.ylabel('True Positive Rate (TPR)',weight='bold')
plt.title('ROC curve',weight='bold')
plt.legend(loc='lower right')
plt.savefig("./1_cnn_append_data_unbalance_auc_curve.png")

# 保存最优的模型
import joblib
best_model_file = "best_model_1_cnn_celoss_append_data_unbalance.pkl"
joblib.dump(model, best_model_file)
print(f"最优模型已保存到 {best_model_file}")