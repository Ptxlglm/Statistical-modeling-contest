# 2. 探索性数据分析(EDA)
# 2.1 数据概览
print("数据形状:", df.shape)
print("\n数据类型:\n", df.dtypes) # 查看数据框中各列的数据类型
print("\n描述性统计（分病例/对照组）:")
print(df.groupby('group(2对照,1病例)').describe().round(2).T)
# groupby()方法按分组列（病例=1，对照=0）拆分数据集
# describe()是Pandas中的一个方法，用于生成数值列的统计量(计数、均值、标准差、最小值、四分位数、中位数和最大值)
# round()函数用于对数值进行四舍五入，保留两位小数(这个描述不准确，但差不多理解就行)
# 方法链式调用————连续使用.describe().round(2)
# 每个方法都有返回对象，df.describe()生成另一个DataFrame，
# 然后调用round(2)对其进行处理

# 默认 df.describe() 仅输出数值列的统计
# df.describe(include='all') 会额外显示分类变量的统计信息，包括非数值列(计数、唯一值数量、出现最频繁的值及其出现次数)

# .T转置DataFrame，将行和列交换。
# 原始输出中，每个统计量是行，分组是列，转置后分组变为行，统计量变为列，更便于阅读。

# 输出结果:
# group(2对照,1病例)       1        2
# count               385.00   420.00
# mean                  3.45     2.18
# std                   1.02     0.87
# min                   1.20     0.80
# 25%                   2.80     1.50
# 50%                   3.40     2.10
# 75%                   4.00     2.80
# max                   6.50     4.20

# 2.2 目标变量分析
plt.figure(figsize=(10,5)) # 设置画布尺寸为10英寸宽×5英寸高
ax = sns.countplot(x='group(2对照,1病例)', data=df) # 创建分组计数柱状图
plt.title('病例组与对照组样本分布') # 设置图表标题
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center')
# ax.annotate() 参数
# f'\n{p.get_height()}',  # 要显示的文本（换行符+柱高）   \n产生换行，使数字显示在柱子顶部上方
# (p.get_x() + 0.2, p.get_height()),  # 文本位置坐标   p.get_x()获取柱子左侧x坐标，+0.2向右偏移；y坐标为柱高
# ha = 'center')  # 水平居中对齐

# sns.countplot() 参数
# 参数	作用	        扩展说明
# x	    指定分组变量	该列应为分类变量（如1=病例，0=对照）
# data	数据来源	    从DataFrame中读取数据

# 在Matplotlib中，axes对象（通常简写为ax）代表一个图表区域，
# 而patches是axes中的一个属性，用于存储所有“补丁”对象。
# 补丁是基本的图形元素，如矩形、圆形等，用于构建更复杂的图形。
# 例如，柱状图中的每个柱子就是一个矩形补丁，
# 饼图中的每个扇形也是一个补丁。
# ax.patches用于直接操作图表中的基本图形元素（如柱状图的柱子、饼图的扇形等）

# 例如，在Seaborn的countplot中，
# 每个柱子对应一个Rectangle对象，存储在ax.patches列表中

# countplot 和 barplot 的柱子都是 Rectangle 对象
# 均存储在 ax.patches 中
# 两者功能定位不同，但图形元素实现方式一致

# 2.3 医学特征分析
# 2.3.1 人口学特征与患病率
demo_cols = ['sex', 'age', 'drink', 'smk']
plt.figure(figsize=(15,8))
for i, col in enumerate(demo_cols, 1):
    plt.subplot(2,2,i)
    if col == 'age': # 连续变量age使用sns.boxplot()方法，画出箱线图比较病例组与对照组的年龄分布差异
        sns.boxplot(x='group(2对照,1病例)', y=col, data=df)
    else: # 分类变量'sex', 'drink', 'smk'使用.plot(kind='bar')，画出患病率柱状图展示不同类别中的疾病发生率
        df.groupby(col)['group(2对照,1病例)'].mean().mul(100).plot(kind='bar')
        plt.ylabel('患病率 (%)')
    plt.title(f'{col}与疾病关系')

# 用图表直观展示不同人群特征与患病风险的关系
# 对比年龄差异
# 怎么做：用箱线图（类似身高体重体检报告中的分布图）展示患者组和健康组的年龄分布。
# 能看出什么：
# 患者组是否普遍年龄更大？
# 两组的年龄范围是否有明显差异？
# 示例：如果患者组的箱子整体右移，说明患病风险可能随年龄增长增加
# 分析性别、烟酒习惯的影响
# 怎么做：用柱状图显示不同人群的患病百分比。
# 能看出什么：
# 男性 vs 女性谁更容易患病？
# 吸烟喝酒的人患病比例是否更高？
# 示例：若吸烟者的柱子明显更高，说明吸烟可能是危险因素
# 整合所有分析到一张大图
# 排版：像拼图一样把四个小图（年龄、性别、饮酒、吸烟）整齐排列。
# 优点：方便一次性对比所有关键因素，避免来回翻页查看。

# 数据分组 df.groupby(col)
# 功能：根据列 col（如性别）将数据分组
# 示例：如果 col='sex'，数据会被分为 男性 和 女性 两组
# 2. 选择目标列 ['group(2对照,1病例)']
# 列含义：假设该列编码为：
# 1 = 病例（患者）
# 2 = 对照（健康人）
# 3. 计算均值 .mean()
# 4. 转换为百分比 .mul(100)
# 5. 绘图 .plot(kind='bar')
# 输出效果：生成柱状图，x轴为分组类别（如男/女），y轴为患病率（%）

# enumerate(demo_cols, 1)
# 功能：为列表中的每个元素生成带编号的配对，类似给物品贴标签
# 参数解释：
# demo_cols：要处理的列表（如 ['性别', '年龄', '饮酒', '吸烟']）
# 1：编号从1开始（默认从0开始）
# 输出示例：
# (1, '性别')
# (2, '年龄')
# (3, '饮酒')
# (4, '吸烟')
# 实际用途：
# 在绘制4张小图时，用编号 i 确定每张图的位置（如 i=1 对应左上角，i=4 对应右下角）
# 2. sns.boxplot(x='group(2对照,1病例)', y=col, data=df)
# 功能：绘制箱线图，对比不同组别的数值分布差异
# 参数详解：
# 参数	                作用	类比说明
# x='group(2对照,1病例)'	分组依据列	就像把学生分成「男生组」和「女生组」
# y=col	                分析的数值列	要比较的指标（如考试成绩）
# data=df	            数据来源	    就像从成绩册中取数据

# 未使用plt.xlabel的原因分析
# 默认标签机制
# Seaborn箱线图自动使用x参数（group）作为x轴标签
# Pandas柱状图自动将分组索引（如性别中的"Male"/"Female"）作为x轴刻度

# 2.3.2 重金属浓度对比（病例vs对照）
metal_cols = ['Zn', 'Ni', 'Cu', 'Pb', 'Fe']
case_metal = df[df['group(2对照,1病例)']==1][metal_cols].mean() # 取病例组数据列个excel表，再取其中的五个重金属那几列，分别求病例组中五种重金属浓度的均值
control_metal = df[df['group(2对照,1病例)']==0][metal_cols].mean() # 同上

plt.figure(figsize=(10,6))
sns.heatmap(pd.concat([case_metal, control_metal], axis=1).T, # 将病例组和对照组数据横向拼接，之后转置
            annot=True, fmt=".1f", cmap='coolwarm', # annot=True	在热力格中显示数值   fmt=".1f"	数值显示格式为保留1位小数   cmap='coolwarm'	使用蓝-红渐变色系	红色表示高浓度（病例组高暴露），蓝色表示低浓度（对照组低暴露）
            yticklabels=['病例组', '对照组']) # yticklabels	自定义Y轴标签	将默认的行索引（如0,1）替换为更易懂的"病例组"、"对照组"
plt.title('重金属浓度均值对比（标准化后）')
# 可视化病例组与对照组的各金属的平均浓度差异
# 拼接后结构：
#          病例组   对照组
# Zn      45.2   38.7
# Ni       2.1    1.5
# Cu      12.3    9.8
# 转置.T
# 效果：行列互换，适应热力图的视觉需求
# 转置后结构：
#         Zn    Ni    Cu
# 病例组  45.2   2.1   12.3
# 对照组  38.7   1.5    9.8

# 2.4 特征交互分析
sns.pairplot(df[['age', 'Zn', 'Ni', 'group(2对照,1病例)']], # 把这几列单独拎出来成一个excel表
             hue='group(2对照,1病例)', plot_kws={'alpha':0.6})
plt.suptitle('关键变量交互关系', y=1.02) # 标题定位：y=1.02 将主标题略微上移，避免与子图标题重叠
# 探索关键变量间的交互关系与组间差异模式，通过散点图矩阵呈现多变量联合分布特征
# hue='group(2对照,1病例)'
# 作用：按病例组/对照组着色
# 可视化效果：
# 病例组数据点显示为一种颜色（如红色）
# 对照组显示为另一种颜色（如蓝色）
# plot_kws={'alpha':0.6}
# 透明度调节：降低点的不透明度（alpha=0.6 → 60%不透明）
# 解决痛点：缓解散点图重叠问题，更清晰显示密集区域

# 2.5 模型驱动分析
# 2.5.1 随机森林特征重要性
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(10,6))
importances.nlargest(10).sort_values(ascending=False).plot(kind='barh', color='darkcyan')
plt.title('随机森林特征重要性（Top 10）')
plt.xlabel('重要性得分')
# 找出哪些特征（如年龄、重金属浓度等）对预测结直肠癌最重要
# rf = RandomForestClassifier(n_estimators=200, random_state=42)
# 创建随机森林分类器
# n_estimators=200：指定森林中决策树的数量为200棵。树越多，模型越稳定，但计算量增加。
# random_state=42：固定随机种子，确保每次运行结果一致（42是常用示例值），便于实验结果复现

# rf.fit(X_train_scaled, y_train)
# X_train_scaled：经过标准化处理的训练集特征数据（如归一化处理后的数值）。
# y_train：训练集的标签数据（预测目标）。
# fit()：训练方法，让模型从数据中学习特征与标签的关系。

# importances = pd.Series(rf.feature_importances_, index=X.columns)
# feature_importances_：随机森林计算的特征重要性得分，表示每个特征对预测的贡献。
# pd.Series：将随机森林计算出的特征重要性数值（rf.feature_importances_），
    # 与特征名称（X.columns）绑定，形成「数值+名称」的对应关系
# index=X.columns
# X.columns：表示数据集 X 的所有特征名称（如 ["年龄", "收入", "性别"]）
# 原始数据：rf.feature_importances_ 是一个纯数值数组，如 [0.2, 0.5, 0.3]
# 添加标签：index=X.columns 把特征名称（例如 ["年龄", "收入", "性别"]）作为标签，绑定到数值上，变成：
# 年龄    0.2
# 收入    0.5
# 性别    0.3

# importances.nlargest(10).sort_values(ascending=False).plot(kind='barh', color='darkcyan')
# 取重要性最高的10个特征并按降序排列，即重要性越高的越往前排
# kind='bar' → 垂直条形图（数值在Y轴，特征名称在X轴）
# kind='barh' → 水平条形图（数值在X轴，特征名称在Y轴）
# color='darkcyan'**
# 作用：设置条形图的颜色为深青色

# 2.5.2 多重共线性检测
vif_data = pd.DataFrame() # 初始化一个空的Pandas DataFrame，用于后续存储特征名称及其对应的方差膨胀系数（VIF）值
vif_data["feature"] = X.columns # 将原始数据集 X 的列名（即特征名称）存入DataFrame的 feature 列<创建这列>
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))] # 遍历每个特征，计算其方差膨胀系数（VIF），并将结果存入DataFrame的 VIF 列
print("\n方差膨胀系数(VIF):\n", vif_data.sort_values('VIF', ascending=False))
# variance_inflation_factor()：用于计算VIF的统计函数，来自 statsmodels 库
# X.values：将数据集 X 转换为NumPy数组格式（二维数组，每行代表一个样本，每列代表一个特征）
# 将DataFrame按 VIF 列降序排列后输出，便于优先查看共线性较高的特征
# 共线性与方差膨胀系数VIF关系？

# 2.5.3 高相关性筛选
corr_matrix = df.corr().abs()
high_corr = corr_matrix[corr_matrix > 0.7].stack().reset_index()
high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]
print("\n高相关特征对（r>0.7）:\n", high_corr.drop_duplicates().sort_values(0, ascending=False))
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
