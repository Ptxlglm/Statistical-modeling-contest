# 1. 数据预处理
# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1.1 读取Excel文件
df = pd.read_excel('重金属.xlsx', sheet_name='Sheet3')

# 1.2 处理目标变量：将group(2对照,1病例)中的2（对照）转为0，1（病例）保持为1
df['group(2对照,1病例)'] = df['group(2对照,1病例)'].replace(2, 0)

# 1.3 处理分类变量：将sex、drink、smk中的2转为0（假设2表示否定/女性，需根据实际含义调整）
for col in ['sex', 'drink', 'smk', 'family']:
    df[col] = df[col].replace(2, 0)

# 1.4 删除无关列：ID列不参与建模(防止这些非特征数据干扰模型训练)
df.drop('ID', axis=1, inplace=True)

# 1.5 检查缺失值
print("缺失值统计：\n", df.isnull().sum())

# 1.6 填充缺失值（假设用中位数填充数值型，众数填充分类型）
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = ['sex', 'drink', 'smk', 'family']

# 数值列处理
for col in numeric_cols:  # 遍历所有数值型列
    if df[col].isnull().sum() > 0:  # 检测是否存在缺失值
        df[col].fillna(df[col].median(), inplace=True)

# 分类列处理
for col in categorical_cols:  # 遍历所有分类型列
    if df[col].isnull().sum() > 0:  # 检测是否存在缺失值
        df[col].fillna(df[col].mode()[0], inplace=True)

# 1.7 划分数据集
X = df.drop('group(2对照,1病例)', axis=1)
y = df['group(2对照,1病例)']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 1.8 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 输出处理后的数据示例
print("\n处理后的训练集维度:", X_train_scaled.shape)
print("处理后的测试集维度:", X_test_scaled.shape)



# 2. 描述性统计分析
# 2.1 数据概况分析
print("数据概况：")
print(df.describe().round(2))  # 数值型变量描述统计，保留2位小数

# 2.2 类别变量分布分析
print("\n类别变量分布：")
for col in ['sex', 'drink', 'smk']:
    print(f"\n{col}分布：")
    print(df[col].value_counts(normalize=True))  # 显示比例分布

# 2.3 目标变量平衡性检查
print("\n病例/对照组比例：")
print(df['group(2对照,1病例)'].value_counts(normalize=True))

# 2.4 相关性初筛（耗时操作，大数据时慎用）
corr_matrix = df[numeric_cols].corr().abs()
high_corr = corr_matrix[corr_matrix > 0.7].stack().reset_index()
high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]  # 排除对角线
print("\n高相关性特征对（r>0.7）：")
print(high_corr.drop_duplicates().sort_values(0, ascending=False))