# 2. 描述性统计分析
# 2.1. 数据概况分析
print("数据概况：")
print(df.describe().round(2))  # 数值型变量描述统计，保留2位小数
# describe()是Pandas中的一个方法，用于生成常用统计量(计数、均值、标准差、最小值、四分位数、中位数和最大值)
# round()函数用于对数值进行四舍五入，保留两位小数(这个描述不准确，但差不多理解就行)
# 方法链式调用————连续使用.describe().round(2)
# 每个方法都有返回对象，df.describe()生成另一个DataFrame，
# 然后调用round(2)对其进行处理

# 2.2. 类别变量分布分析
print("\n类别变量分布：")
for col in ['sex', 'drink', 'smk']:
    print(f"\n{col}分布：")
    print(df[col].value_counts(normalize=True))  # 显示比例分布
# f-string格式化字符串，打印当前列的分布标题
# 确保每个列的分布信息之间有一个空行，使输出更清晰
# {col}会被替换为当前的列名，例如第一次循环时显示“sex分布：”
# .value_counts(normalize=True)：
# 这是Pandas的Series方法，用于计算该列中每个唯一值的出现次数
# 参数normalize=True表示
# 返回的是比例而非绝对计数（即各值出现的频率，总和为1）
# 输出结果形式如下：
# sex分布：
# 1    0.75
# 0    0.25
# Name: sex, dtype: float64
#
# drink分布：
# 0    0.5
# 1    0.5
# Name: drink, dtype: float64
#
# smk分布：
# 1    0.5
# 0    0.5
# Name: smk, dtype: float64

# 2.3. 目标变量平衡性检查
print("\n病例/对照组比例：")
print(df['group(2对照,1病例)'].value_counts(normalize=True))
# 代码意思同上
# 代码意义如下：
# 验证是否严重不平衡（如病例:对照=1:9时需要采样处理），这对所有分类模型都至关重要

# 2.4. 相关性初筛（耗时操作，大数据时慎用）
corr_matrix = df[numeric_cols].corr().abs()
high_corr = corr_matrix[corr_matrix > 0.7].stack().reset_index()
high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]  # 排除对角线
print("\n高相关性特征对（r>0.7）：")
print(high_corr.drop_duplicates().sort_values(0, ascending=False))
# 第一句————计算相关系数矩阵
# df[numeric_cols]：筛选数值型的列（排除性别等分类变量），numeric_cols应该是一个包含数值型列名的列表。(前无可？)
# .corr()：计算这些数值型列之间的相关系数矩阵，通常是皮尔逊相关系数。
# .abs()：取相关系数的绝对值，因为无论是正相关还是负相关，只要绝对值大，我们都关心。
# 所以这一行的结果是得到一个绝对值后的相关系数矩阵，所有值都在0到1之间。
# 第二句————筛选高相关性的部分
# corr_matrix > 0.7：生成一个布尔矩阵，其中True表示相关系数绝对值大于0.7的位置。
# corr_matrix[corr_matrix > 0.7]：用布尔矩阵筛选出相关系数大于0.7的值，其他位置变为NaN。
# .stack()（n一摞v叠成一摞）：将矩阵从二维的DataFrame转换为一维的Series，其中每行由原来的行索引、列索引和值组成。
# 这样每个高相关的特征对都会被展开成一行。
# 将矩阵转换为长格式（行索引、列索引、数值三元组）
# .reset_index()：将索引转换为列，生成一个包含'level_0'（原行索引）、'level_1'（原列索引）和0（相关系数值）的DataFrame。
# 就是相当于生成一个列表形式而已
# 第三句————进一步去除没必要的
# 排除对角线上的元素，因为特征与自身的相关系数总是1，这不具有分析意义。
# 这里通过比较'level_0'和'level_1'是否不同来过滤掉这些情况。
# 第五句
# drop_duplicates()（多重记录、复制品）：利用矩阵对称性去重，去除重复的行。由于相关系数矩阵是对称的（如特征A与特征B的相关性和特征B与特征A的相关性相同），这里可能存在重复记录。
# sort_values(0, ascending=False)（v升序）：按相关系数值（列名为0）降序排序，使最高的相关性排在最前面。
# 最后打印处理后的结果，显示高相关的特征对及其相关系数。

# 相关系数矩阵 → 筛选高相关 → 矩阵转置 → 去重排序
# (热力图)     (阈值过滤)   (数据重塑)   (结果优化)