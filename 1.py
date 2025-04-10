机器学习的随机森林 XGboost 支持向量机 LASSO WQS回归模型

贝叶斯核函数回归模型

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_excel("重金属.xlsx", sheet_name="Sheet3")

# 重命名列名为target(对象)
df = df.rename(columns={'group(2对照，1病例)': 'target'})

# 1. 数据预处理
def preprocess_data(df):
    # 处理缺失值（假设用中位数填充）
    # df.fillna(df.median(), inplace=True)
    df.fillna(None, inplace=True)

    # 分类变量编码（如sex, drink等已经是0/1格式则无需处理）
    # 若存在多分类变量，可使用pd.get_dummies

    # 验证列名
    print("当前列名:", df.columns)

    # 分离特征与目标变量
    X = df.drop(columns=['ID', 'target'])
    y = df['target']
    # group: 0=对照, 1=病例
    y = y.replace({2: 0, 1: 1})

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = preprocess_data(df)

# 2. 描述性统计分析
print("病例组比例:", y_train.mean())
print("\n重金属浓度统计:")
print(X_train.filter(regex='Zn|Ni|Pb|Cu').describe())


# 3. LASSO回归特征选择
def lasso_feature_selection(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用Lasso进行特征选择
    lasso = Lasso(alpha=0.01)  # 调节alpha控制稀疏性
    lasso.fit(X_scaled, y)

    # 筛选非零系数特征
    selected_features = X.columns[lasso.coef_ != 0]
    print("\nLASSO选中的特征:", selected_features.tolist())

    return selected_features


selected_features = lasso_feature_selection(X_train, y_train)


# 4. WQS回归（加权分位数和）
def wqs_regression(X, y, n_quantiles=4):
    # 分位数化暴露变量（以重金属为例）
    metals = ['Zn', 'Ni', 'Hf', 'Mg', 'Ti', 'V', 'Co', 'As', 'Se', 'Sr', 'Sn', 'Sb', 'Cs', 'La', 'Lu', 'Tl', 'Pb', 'Fe',
              'Mo', 'Cu', 'Rb', 'Ba', 'Ir']
    X_metals = X[metals]

    # 分箱处理
    quantile = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform')
    X_quantiled = pd.DataFrame(quantile.fit_transform(X_metals), columns=metals)

    # 计算分位数指数
    X_quantiled['wqs_index'] = X_quantiled.sum(axis=1)  # 实际需迭代优化权重

    # 合并其他变量
    X_wqs = X.drop(columns=metals).join(X_quantiled['wqs_index'])

    # 逻辑回归
    model = LogisticRegression()
    model.fit(X_wqs, y)
    print("\nWQS模型系数:", model.coef_)

    return model


wqs_model = wqs_regression(X_train, y_train)


# 5. 机器学习模型（以随机森林为例）
def train_ml_model(X, y):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 交叉验证评估
    scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
    print("\n随机森林平均AUC:", scores.mean())

    # 训练最终模型
    pipe.fit(X, y)
    return pipe


ml_model = train_ml_model(X_train[selected_features], y_train)


# 6. 模型评估（AUC/ROC）
def evaluate_model(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    return auc


print("\n随机森林测试集AUC:", evaluate_model(ml_model, X_test[selected_features], y_test))

# 7. BKMR替代方案（高斯过程回归）
import GPy


def bkmr_analysis(X, y):
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用高斯过程回归
    kernel = GPy.kern.RBF(input_dim=X_scaled.shape[1])
    model = GPy.models.GPClassification(X_scaled, y.values.reshape(-1, 1), kernel=kernel)
    model.optimize(messages=True)

    # 可视化核权重（近似变量重要性）
    print("\nRBF核长度尺度:", model.kern.lengthscale.values)


bkmr_analysis(X_train[selected_features], y_train)