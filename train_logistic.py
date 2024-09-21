import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 假设你的特征数据保存在X中，标签保存在y中
excel_file = "OPLL-机器学习-C5p患者资料.xlsx"
sheet_name = "append_data_unbalance"
df = pd.read_excel(excel_file, sheet_name=sheet_name)
X = df.iloc[:, 1:]
X = X.fillna(-99)
print(X.columns)

y = df.iloc[:, 0]

min_max_scaler = preprocessing.MinMaxScaler()  
X_minMax = min_max_scaler.fit_transform(X)

X_minMax = np.array(X_minMax)# 你的特征数据 (240x32的矩阵)
y = np.array(y)# 你的标签数据 (240个元素的向量)
print(X_minMax)


# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_minMax, y, test_size=0.2, random_state=42)

# 创建多元逻辑回归模型
model = LogisticRegression(solver='lbfgs', class_weight='balanced')

# 创建一个分层K折交叉验证对象
cv = StratifiedKFold(n_splits=5)

# 设置参数网格，可以根据需要调整
param_grid = {
    'C': [0.01, 0.1, 1, 2, 5, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'max_iter': [50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000],
    'class_weight': ['balanced']
}

# 初始化计数器
eval_counter = 0

# 迭代训练
for train_index, val_index in cv.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # 训练模型
    grid_search = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=3), scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train_fold, y_train_fold)

    # 每隔5轮，在验证集上评估模型
    eval_counter += 1
    if eval_counter % 5 == 0:
        y_val_pred = grid_search.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, y_val_pred)
        precision = precision_score(y_val_fold, y_val_pred)
        recall = recall_score(y_val_fold, y_val_pred)
        f1 = f1_score(y_val_fold, y_val_pred)
        auc = roc_auc_score(y_val_fold, y_val_pred)

        print(f"Evaluation {eval_counter}:")
        print(f"准确率: {accuracy}")
        print(f"精确度: {precision}")
        print(f"召回率: {recall}")
        print(f"F1分数: {f1}")
        print(f"AUC分数: {auc}")

# 获取最佳模型
best_model = grid_search.best_estimator_

# 在测试集上评估最佳模型
y_test_pred = best_model.predict(X_test)
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

# 获取多元逻辑回归模型的系数
coefficients = best_model.coef_

# 计算特征的重要性分数
feature_importance = np.abs(coefficients)

# 获取每个特征的平均重要性分数
average_importance = np.mean(feature_importance, axis=0)

# 排序特征索引，按照平均重要性分数从高到低
sorted_feature_indices = np.argsort(average_importance)[::-1]


import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
plt.rcParams['axes.unicode_minus']=False #显示负号

coef_lr = pd.DataFrame({'var' : X.columns,
                        'coef' : best_model.coef_.flatten()
                        })

index_sort =  np.abs(coef_lr['coef']).sort_values().index
coef_lr_sort = coef_lr.loc[index_sort,:]

fig,ax=plt.subplots()
x, y = coef_lr_sort['var'], coef_lr_sort['coef']
rects = plt.barh(x, y, color='chocolate')
plt.grid(linestyle="-.", axis='y', alpha=0.4)
plt.tight_layout()
#添加数据标签
for rect in rects:
    w = rect.get_width()
    ax.text(w, rect.get_y()+rect.get_height()/2,'%.2f' %w,ha='left',va='center')

plt.savefig("./logistic_append_data_unbalance_important_feature.png")


import seaborn as sns
df_coor = df.iloc[:, 1:].corr()

fig, ax = plt.subplots(figsize=(16, 16),facecolor='w')
# 指定颜色带的色系
sns.heatmap(df.iloc[:, 1:].corr(),annot=True, vmax=1, square=True, cmap="Blues", fmt='.2g')
plt.title('Correlation heatmap',weight='bold')
plt.show()

fig.savefig('./df_corr.png',bbox_inches='tight',transparent=True)



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
plt.savefig("./logistic_append_data_unbalance_auc_curve.png")



# 保存最优的模型
best_model_file = "best_model_logistic_append_data_unbalance.pkl"
joblib.dump(best_model, best_model_file)
print(f"最优模型已保存到 {best_model_file}")