import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 假设你的特征数据保存在X中，标签保存在y中
excel_file = "OPLL-机器学习-C5p患者资料.xlsx"
sheet_name = "append_data_unbalance"
df = pd.read_excel(excel_file, sheet_name=sheet_name)
#excel_file = "opll_data_unbalance.csv"
#df = pd.read_csv(excel_file, encoding='gbk')
X = df.iloc[:, 1:]
X = X.fillna(-99)
y = df.iloc[:, 0]

X = np.array(X)# 你的特征数据 (240x32的矩阵)
y = np.array(y)# 你的标签数据 (240个元素的向量)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建XGBoost分类器
model = xgb.XGBClassifier()

# 创建一个分层K折交叉验证对象
cv = StratifiedKFold(n_splits=5)

param_grid = {
    'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200],
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4, 5, 6, 7],
    'min_child_weight': [2, 3, 4, 5, 6, 7],
    'scale_pos_weight': [2, 3, 4, 5, 6, 7],
    'objective': ['binary:logistic'],
    'subsample':[0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1],
    'gamma':[0.1, 0.5, 1],
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
feature_names = df.iloc[:, 1:].columns#[f'Feature_{i}' for i in range(X.shape[1])]

# 创建特征重要性字典，将特征名和重要性分数对应起来
feature_importance_dict = dict(zip(feature_names, importance_scores))

# 排序特征重要性，按照重要性分数从高到低
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=False)


import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
plt.rcParams['axes.unicode_minus']=False #显示负号
sorted_feature_names, sorted_importance_scores = zip(*sorted_feature_importance)

plt.figure(figsize=(16, 12))
fig,ax=plt.subplots()
x, y = sorted_feature_names, sorted_importance_scores
rects = plt.barh(x, y, color='chocolate')
plt.grid(linestyle="-.", axis='y', alpha=0.4)
plt.tight_layout()
#添加数据标签
for rect in rects:
    w = rect.get_width()
    ax.text(w, rect.get_y()+rect.get_height()/2,'%.2f' %w,ha='left',va='center')
plt.savefig("./xgb_append_data_unbalance_important_feature.png")


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, str(feat)))
        i = i + 1
    outfile.close()

ceate_feature_map(df.iloc[:, 1:].columns)

xgb.plot_tree(best_model, num_trees=0, fmap='xgb.fmap')
fig = plt.gcf()
fig.set_size_inches(150, 100)
#plt.show()
fig.savefig('./xgb_append_data_unbalance_decision_tree1.png')



from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

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
plt.savefig("./xgb_append_data_unbalance_auc_curve.png")


import joblib
# 保存最优的模型
best_model_file = "best_model_xgb_append_data_unbalance.pkl"
joblib.dump(best_model, best_model_file)
print(f"最优模型已保存到 {best_model_file}")