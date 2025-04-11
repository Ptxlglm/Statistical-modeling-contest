# 数据预处理
# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 读取Excel文件
df = pd.read_excel('重金属.xlsx', sheet_name='Sheet3')
# pandas中的read_excel()方法————>功能：专门解析Excel文件的函数，将表格结构转换为内存中的DataFrame对象
# DataFrame的概念，类似于Excel表格，有行和列的结构，方便后续处理
# 返回值df的数据类型：pandas.DataFrame

# 2. 处理目标变量：将group(2对照,1病例)中的2（对照）转为0，1（病例）保持为1
df['group(2对照,1病例)'] = df['group(2对照,1病例)'].replace(2, 0)
# Pandas的操作通常不是原地修改(Pandas默认返回新对象)
# 所以需要赋值，即返回修改后的新Series对象

# 3. 处理分类变量：将sex、drink、smk中的2转为0（假设2表示否定/女性，需根据实际含义调整）
for col in ['sex', 'drink', 'smk', 'family']:
    df[col] = df[col].replace(2, 0)
# 针对[]里的每个指定列，将2替换为0
# 为什么不合并group(2对照,1病例)这一列

# 4. 删除无关列：ID列不参与建模(防止这些非特征数据干扰模型训练)
df.drop('ID', axis=1, inplace=True)
# df.drop()功能：删除行或列
# axis(操作轴方向：0=行，1=列)
# inplace参数(是否原地修改)的作用是直接修改原DataFrame(=True)，
# 而不是返回一个新的副本，这样避免了重新赋值的麻烦，
# 但需要注意inplace的使用可能带来的问题，比如数据被意外修改

# 5. 检查缺失值
print("缺失值统计：\n", df.isnull().sum())
# df.isnull()方法的作用是生成一个布尔类型的DataFrame，
# 其中每个元素表示原数据中的对应位置是否为缺失值（NaN或None）
# .sum()方法会对这些布尔值进行求和，
# 因为True被视为1，False为0，所以每列的缺失值数量会被计算出来

# 6. 填充缺失值（假设用中位数填充数值型，众数填充分类型）
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = ['sex', 'drink', 'smk', 'family']
# 第一句————自动识别数值型列
# 通过select_dtypes方法
# 筛选DataFrame中所有数值型（整数、浮点数等）列，生成列名列表
# 保存到列表numeric_cols中
# 内存变化示意图
#     A[原始DataFrame] --> B[select_dtypes筛选]
#     B --> C[仅含数值型列的新DF]
#     C --> D[提取列名索引]
#     D --> E[转换为列表]
# 第二句————手动指定分类型列
# 手动指定需要作为分类变量处理的列名
# 几个需要特殊处理的栏目
# 这些栏目里的数字其实是代表“类型”
# (这些列虽以数字形式存储（如0/1），但实际代表离散类别，需特殊编码（如OneHot Encoding）<独热编码>)
# 避免机器学习模型将分类变量误判为连续变量（例如认为性别2比性别1“大”

# 对比说明（为什么分开处理？）
# 特征类型	示例列	数据处理需求
# 数值型	年龄、血压	标准化、处理异常值
# 分类型	性别、吸烟	独热编码、检查类别平衡性
# 通过这种分离，可为后续的统计建模（如线性回归需要数值型输入）和机器学习（如随机森林需编码分类变量）提供结构化的特征输入。

# 数值列处理
for col in numeric_cols:  # 遍历所有数值型列
    if df[col].isnull().sum() > 0:  # 检测是否存在缺失值
        # 用中位数median填充缺失
        # 数学原理：中位数 = 排序后50%分位的值
        # 适用场景：适合存在偏态分布的数值数据（如收入数据）
        # 抗噪性：不受极端值影响（若数据中有极大 / 极小异常值，中位数比均值更稳定）
        df[col].fillna(df[col].median(), inplace=True)
        # fillna() 是 pandas 中用于填充缺失值（NaN）的核心方法，
        # 适用于 DataFrame 和 Series
        # 用列均值填充且直接修改————df[col].fillna(df[col].median(), inplace=True)
        # 所有NaN替换为0————df.fillna(0)

# 分类列处理
for col in categorical_cols:  # 遍历所有分类型列
    if df[col].isnull().sum() > 0:  # 检测是否存在缺失值
        # 用众数mode（第一众数）填充缺失
        # mode()[0]的含义：
        # mode()返回众数的Series，
        # [0]取第一个众数（当存在多个众数时）
        df[col].fillna(df[col].mode()[0], inplace=True)

# 7. 划分数据集
X = df.drop('group(2对照,1病例)', axis=1)
y = df['group(2对照,1病例)']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
# df.drop()：Pandas DataFrame的列删除方法
# 'group(2对照,1病例)'：要删除的目标列名
# axis=1：指定操作方向为列（axis=0是行操作）
# 创建特征矩阵X，移除原始数据中的目标变量列（'group(2对照,1病例)'）
# 相当于：X = df[所有列] - 'group(2对照,1病例)'列
# 输出数据类型：pandas DataFrame（保持多列结构）

# df['group(2对照,1病例)']：单列选择语法
# 无参数方法直接提取列数据
# 创建目标变量y，即需要预测的标签列
# 输出数据类型：pandas Series（单列结构）

# 参数详解：
#     X：特征矩阵（所有输入变量）
#     y：目标变量（预测标签）
#     test_size=0.3：测试集比例30%（训练集自动70%）
#     stratify=y：分层抽样，保持y的类别分布
#     random_state=42：随机种子，保证可重复性

# 数据分割原理：
#     原始数据行数：1000
#     分割后：
#     X_train.shape = (700, 特征数)
#     X_test.shape = (300, 特征数)
#     stratify关键作用：
#
# 当y为不平衡分类数据（如病例:对照=1:9）时：
# 强制保持训练/测试集中该比例一致
# 避免随机分割导致子集比例偏移
# random_state机制：
#
# 指定伪随机数生成器的种子
# 相同种子保证每次分割结果相同
# 重要用途：实验可复现性

# 输出对象说明：
#     变量名	数据类型	作用
#     X_train	DataFrame	训练集特征数据
#     X_test	DataFrame	测试集特征数据
#     y_train	Series	训练集标签
#     y_test	Series	测试集标签

# 8. 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 第一句————创建标准化器
# StandardScaler是来自sklearn.preprocessing模块的一个类，
# 用于标准化数据。
# scaler是一个StandardScaler的实例，
# fit_transform是它的方法，
# 接受X_train作为参数
# StandardScaler()：实例化一个标准化处理器对象

# 创建scaler实例后，fit_transform方法用于拟合数据并转换训练集，
# 而transform方法则应用同样的参数转换测试集

# 机制
# 初始化一个未训练的标准化器
# 内存中创建对象存储空间，准备接收数据

# 作用
# 对训练集和测试集进行标准化处理，
# 确保数据适合模型训练，并保持评估的公正性

# 第二句————训练并转换训练集
# 语法解析：
# **.fit()**：
# 计算训练数据的均值（μ）和标准差（σ）
# 数学公式：
# μ = np.mean(X_train, axis=0)
# σ = np.std(X_train, axis=0)
# 存储计算结果到scaler对象的属性中：
# scaler.mean_     # 均值向量
# scaler.scale_    # 标准差向量
# **.transform()**：
# 应用标准化公式转换数据：
# X_scaled = (X - μ) / σ
# 输出为NumPy数组（维度与输入一致）

# 技术细节：
# fit_transform() 是 fit() + transform() 的快捷方法
# 训练集必须同时执行拟合和转换
# # 转换后数据满足：
# np.mean(X_train_scaled, axis=0) ≈ 0   # 各列均值趋近0
# np.std(X_train_scaled, axis=0) ≈ 1    # 各列标准差趋近1

# 第三句————转换测试集
# X_test_scaled = scaler.transform(X_test)
# 关键要点：
# 禁止重新拟合：直接使用训练集的μ和σ
    # 测试集必须使用训练集的参数，以避免数据泄漏，确保模型评估的准确性
    # 必要性：防止数据泄露（Data Leakage）
# 数学运算：
# X_test_scaled = (X_test - scaler.mean_) / scaler.scale_

# 输出数据结构
# 变量名	数据类型	维度	数值特性
# X_train_scaled	NumPy ndarray	(n_samples, n_features)	各列均值≈0，标准差≈1
# X_test_scaled	NumPy ndarray	(m_samples, n_features)	同训练集缩放参数

# 输出处理后的数据示例
print("\n处理后的训练集维度:", X_train_scaled.shape)
print("处理后的测试集维度:", X_test_scaled.shape)
# shape属性：NumPy数组/Pandas DataFrame的固定属性，
# 返回数据维度元组 (n_samples, n_features)