import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

# 创建随机森林分类器
model = RandomForestClassifier()

# 创建一个分层K折交叉验证对象
cv = StratifiedKFold(n_splits=5)

# 定义超参数网格，包括随机森林的相关超参数
param_grid = {
    'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200],  # 调整估计器的数量
    'max_depth': [2, 3, 4, 5, 6, 7, 8], # 调整树的深度
    'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8],  # 调整节点分裂所需的最小样本数
    'min_samples_leaf': [1, 4, 8, 16, 32, 64],  # 调整叶节点的最小样本数
    'max_features': ['auto', 'sqrt', 'log2'],  # 调整每个分裂考虑的最大特征数量
    'class_weight': ['balanced'],  # 调整类别权重处理
    'criterion':['gini','entropy'],
    'max_samples':[0.6, 0.7, 0.8, 0.9],
    'n_jobs':[-1]
}


#param_grid = {
#    'n_estimators': [5],  # 调整估计器的数量
#    'max_depth': [2, 3], # 调整树的深度
#    'min_samples_split': [7, 8],  # 调整节点分裂所需的最小样本数
#    'min_samples_leaf': [32],  # 调整叶节点的最小样本数
#    'max_features': ['sqrt', 'log2'],  # 调整每个分裂考虑的最大特征数量
#    'class_weight': ['balanced'],  # 调整类别权重处理
#    'criterion':['gini','entropy'],
#    'max_samples':[0.9],
#    'n_jobs':[-1]
#}

# 使用GridSearchCV来搜索最佳超参数组合
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=cv, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 在测试集上评估最佳模型，包括AUC
y_test_pred = best_model.predict(X_test)
y_test_prob = best_model.predict_proba(X_test)[:, 1]  # 获取正类别的概率

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_prob)  # 计算AUC

print(y_test_pred)
print("在测试集上的性能评估:")
print(f"准确率: {test_accuracy}")
print(f"精确度: {test_precision}")
print(f"召回率: {test_recall}")
print(f"F1分数: {test_f1}")
print(f"AUC: {test_auc}")

# 获取特征重要性
importance_scores = best_model.feature_importances_

# 获取特征名称或索引
#feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
feature_names = df.iloc[:, 1:].columns

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
plt.savefig("./rf_append_data_unbalance_important_feature.png")


from sklearn.tree import export_graphviz
import graphviz
class_names = ["NO", "YES"]
tree = best_model.estimators_[0]
dot_data = export_graphviz(tree, out_file=None, 
                           feature_names=sorted_feature_names,  # 替换为你的特征名称
                           class_names=class_names,  # 替换为你的类别名称
                           filled=True, rounded=True,
                           special_characters=True)

# 生成决策树的图形
graph = graphviz.Source(dot_data.replace('helvetica', 'Cambria'))
graph.format = 'png'
graph.render("rf_append_data_unbalance_decision_tree1")  # 保存图形到文件（可选）




from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)

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
plt.savefig("./rf_append_data_unbalance_auc_curve.png")




# 保存最优的模型
best_model_file = "best_model_rf_append_data_unbalance.pkl"
joblib.dump(best_model, best_model_file)
print(f"最优模型已保存到 {best_model_file}")
