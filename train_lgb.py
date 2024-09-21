import numpy as np
import lightgbm as lgb

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 假设你的特征数据保存在X中，标签保存在y中
excel_file = "OPLL-机器学习-C5p患者资料.xlsx"
sheet_name = "append_data_unbalance"
df = pd.read_excel(excel_file, sheet_name=sheet_name)
X = df.iloc[:, 1:]
X = X.fillna(-99)
y = df.iloc[:, 0]

X = np.array(X)# 你的特征数据 (240x32的矩阵)
y = np.array(y)# 你的标签数据 (240个元素的向量)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建XGBoost分类器
model = lgb.LGBMClassifier()

# 创建一个分层K折交叉验证对象
cv = StratifiedKFold(n_splits=5)

# 定义超参数网格
#param_grid = {
#    'objective':['binary'],
#    'isunbalance':['True'],
#    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
#    'scale_pos_weight': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
#    'mertic':['auc'],
#    'min_child_samples':[15, 20, 25, 30],
#    'max_depth':[2, 3, 4, 5, 6, 7, 8, 9],
#    'num_leaves':[15, 31],
#    'feature_fraction':[0.6, 0.7, 0.8, 0.9, 1],
#    'bagging_fraction':[0.6, 0.7, 0.8, 0.9, 1],
#    'lambda_l1':[0, 0.01, 0.1, 1],
#    'min_child_weight':[1, 3, 5, 7],
#    'verbose':[-1]
#}


# 定义超参数网格
param_grid = {
    'objective':['binary'],
    'isunbalance':['True'],
    'learning_rate': [0.01, 0.1],
    'scale_pos_weight': [3, 4],
    'mertic':['auc'],
    'min_child_samples':[15, 20],
    'max_depth':[8, 9],
    'num_leaves':[15, 31],
    'feature_fraction':[0.9, 1],
    'bagging_fraction':[0.9, 1],
    'lambda_l1':[0.1, 0.5],
    'min_child_weight':[3, 5],
    'verbose':[-1],
    'n_jobs': [-1]
}

# 使用GridSearchCV来搜索最佳超参数组合
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=cv, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 在测试集上评估最佳模型
y_test_pred = best_model.predict(X_test)
print(y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_pred)

print("在测试集上的性能评估:")
print(f"准确率: {test_accuracy}")
print(f"精确度: {test_precision}")
print(f"召回率: {test_recall}")
print(f"F1分数: {test_f1}")
print(f"AUC分数: {test_auc}")

# 获取特征重要性
importance_scores = best_model.feature_importances_

# 获取特征名称或索引
#feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
feature_names = df.iloc[:, 1:].columns

# 创建特征重要性字典，将特征名和重要性分数对应起来
feature_importance_dict = dict(zip(feature_names, importance_scores))

# 排序特征重要性，按照重要性分数从高到低
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# 打印排名前N个重要性最高的特征
N = 10  # 你可以根据需要选择前N个特征
print(f"排名前{N}的重要性特征：")
for i in range(N):
    print(f"特征: {sorted_feature_importance[i][0]}, 重要性分数 = {sorted_feature_importance[i][1]}")

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文宋体
plt.rcParams['axes.unicode_minus']=False #显示负号
sorted_feature_names, sorted_importance_scores = zip(*sorted_feature_importance)

plt.figure(figsize=(16, 12))
fig,ax=plt.subplots()
x, y = sorted_feature_names, sorted_importance_scores
rects = plt.barh(x, y, color='blue')
plt.grid(linestyle="-.", axis='y', alpha=0.4)
plt.tight_layout()
#添加数据标签
for rect in rects:
    w = rect.get_width()
    ax.text(w, rect.get_y()+rect.get_height()/2,'%.2f' %w,ha='left',va='center')
plt.savefig("./lgb_append_data_unbalance.png")
