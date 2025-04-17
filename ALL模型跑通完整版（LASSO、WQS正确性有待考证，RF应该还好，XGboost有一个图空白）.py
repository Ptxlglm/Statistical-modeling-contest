# 1. 数据预处理
# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# 添加全局设置（所有图表生效）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体（黑体）
plt.rcParams['font.family'] = 'sans-serif'    # 全局字体类型
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示乱码

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

df['Cu_Zn_interaction'] = df['Cu'] * df['Zn']
# 添加交互项（如重金属间的协同效应）提高测试集R²

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



# 2. 探索性数据分析(EDA)
# 2.1 数据概览
print("数据形状:", df.shape)
print("\n数据类型:\n", df.dtypes) # 查看数据框中各列的数据类型
print("\n描述性统计（分病例/对照组）:")
print(df.groupby('group(2对照,1病例)').describe().round(2).T)

# 2.2 目标变量分析
plt.figure(figsize=(10,5)) # 设置画布尺寸为10英寸宽×5英寸高
ax = sns.countplot(x='group(2对照,1病例)', data=df) # 创建分组计数柱状图
plt.title('病例组与对照组样本分布') # 设置图表标题
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center')

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

# 2.3.2 重金属浓度对比（病例vs对照）
metal_cols = ['Zn', 'Ni', 'Cu', 'Pb', 'Fe']
case_metal = df[df['group(2对照,1病例)']==1][metal_cols].mean() # 取病例组数据列个excel表，再取其中的五个重金属那几列，分别求病例组中五种重金属浓度的均值
control_metal = df[df['group(2对照,1病例)']==0][metal_cols].mean() # 同上

plt.figure(figsize=(10,6))
sns.heatmap(pd.concat([case_metal, control_metal], axis=1).T, # 将病例组和对照组数据横向拼接，之后转置
            annot=True, fmt=".1f", cmap='coolwarm', # annot=True	在热力格中显示数值   fmt=".1f"	数值显示格式为保留1位小数   cmap='coolwarm'	使用蓝-红渐变色系	红色表示高浓度（病例组高暴露），蓝色表示低浓度（对照组低暴露）
            yticklabels=['病例组', '对照组']) # yticklabels	自定义Y轴标签	将默认的行索引（如0,1）替换为更易懂的"病例组"、"对照组"
plt.title('重金属浓度均值对比（标准化后）')

# 2.4 特征交互分析
sns.pairplot(df[['age', 'Zn', 'Ni', 'group(2对照,1病例)']], # 把这几列单独拎出来成一个excel表
             hue='group(2对照,1病例)', plot_kws={'alpha':0.6})
plt.suptitle('关键变量交互关系', y=1.02) # 标题定位：y=1.02 将主标题略微上移，避免与子图标题重叠

# 2.5 模型驱动分析
# 2.5.1 随机森林特征重要性
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(10,6))
importances.nlargest(10).sort_values(ascending=False).plot(kind='barh', color='darkcyan')
plt.title('随机森林特征重要性（Top 10）')
plt.xlabel('重要性得分')

# 2.5.2 多重共线性检测
vif_data = pd.DataFrame() # 初始化一个空的Pandas DataFrame，用于后续存储特征名称及其对应的方差膨胀系数（VIF）值
vif_data["feature"] = X.columns # 将原始数据集 X 的列名（即特征名称）存入DataFrame的 feature 列<创建这列>
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))] # 遍历每个特征，计算其方差膨胀系数（VIF），并将结果存入DataFrame的 VIF 列
print("\n方差膨胀系数(VIF):\n", vif_data.sort_values('VIF', ascending=False))

# 2.5.3 高相关性筛选
corr_matrix = df.corr().abs()
high_corr = corr_matrix[corr_matrix > 0.7].stack().reset_index()
high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]
print("\n高相关特征对（r>0.7）:\n", high_corr.drop_duplicates().sort_values(0, ascending=False))



# 3. LASSO回归模型构建与特征选择
# 3.1 初始化LASSO回归模型（自动交叉验证选择最优alpha）
lasso = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10],  # 正则化强度候选值
                cv=5,  # 5折交叉验证
                max_iter=10000,  # 最大迭代次数，确保收敛
                random_state=42,  # 随机种子，便于实验复现
                n_jobs=-1)  # 使用全部CPU核心，并行计算加速

# 3.2 训练模型
lasso.fit(X_train_scaled, y_train)

# 3.3 获取特征系数
feature_coef = pd.DataFrame({  # 创建一个excel表
    'Feature': X.columns,  # X.columns：表示数据集 X 的所有特征名称（如 ["年龄", "收入", "性别"]）
    'Coefficient': lasso.coef_,  # LASSO回归模型的特征系数值（包含正负），反映每个特征对目标变量（如结直肠癌发病风险）的影响方向和强度
    'Absolute_Coef': abs(lasso.coef_)  # 系数的绝对值（衡量重要性）
}).sort_values('Absolute_Coef', ascending=False)  # 按绝对值降序排序

# 3.4 筛选重要特征（系数非零的特征）
selected_features = feature_coef[feature_coef['Coefficient'] != 0]['Feature'].tolist()
print("\nLASSO选择的重要特征（非零系数）:")
print(selected_features)

# 3.5 模型评估
# 训练集预测
y_train_pred = lasso.predict(X_train_scaled)
# 测试集预测
y_test_pred = lasso.predict(X_test_scaled)

# 计算评估指标
train_r2 = r2_score(y_train, y_train_pred)  # 训练集R²：模型对训练数据变异的解释能力（0~1，越接近1越好）
test_r2 = r2_score(y_test, y_test_pred)  # 测试集R²：模型对新样本的泛化能力（0~1，越接近1越好）
mse = mean_squared_error(y_test, y_test_pred)  # 均方误差
mae = mean_absolute_error(y_test, y_test_pred)  # 平均绝对误差

# 输出评估结果
print(f"\n最优alpha值: {lasso.alpha_:.4f}")
print(f"训练集R²: {train_r2:.3f}")
print(f"测试集R²: {test_r2:.3f}")
print(f"测试集MSE: {mse:.3f}")
print(f"测试集MAE: {mae:.3f}")

# 3.6 可视化
plt.figure(figsize=(15, 6))

# 系数权重图(第一个子图)
plt.subplot(1, 2, 1)  # 在画布上创建1行2列的子图布局，当前操作第1个子图（左半部分）
coef_plot = feature_coef[feature_coef['Coefficient'] != 0].sort_values('Coefficient')  # 默认升序   按系数值升序排序（负值在左，正值在右）
plt.barh(coef_plot['Feature'], coef_plot['Coefficient'], color='steelblue')  # coef_plot['Feature']	Y轴标签（特征名称），coef_plot['Coefficient']	X轴数值（效应强度）
plt.axvline(0, color='black', linestyle='--', linewidth=1)  # 垂直基线   # 添加参考线
plt.title('LASSO回归特征系数权重', fontsize=14, pad=20)
plt.xlabel('系数值', fontsize=12, labelpad=10)  # labelpad调整标签与轴的距离

# 预测效果散点图(第二个子图)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolor='k')  # 横坐标数据：测试集真实值   纵坐标数据：模型预测值   点透明度alpha最好在0.3~0.8之间	避免重叠点掩盖分布密度，0.6 平衡可见性与重叠显示
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('真实值 vs 预测值')
plt.xlabel('真实值')
plt.ylabel('预测值')

plt.tight_layout()  # 自动调整子图间距，防止标签重叠
plt.show()

# 3.7 保存重要特征结果（供后续建模使用）
selected_features_df = pd.DataFrame({'Selected_Features': selected_features})  # 创建特征列表的DataFrame
selected_features_df.to_csv('selected_features_lasso.csv', index=False)  # 保存到CSV文件（无行索引）



# 4. WQS回归模型构建与混合物效应分析
# 4.1 数据准备
# 根据前面预处理后的数据
metal_cols = ['Zn', 'Ni', 'Cu', 'Pb', 'Fe']  # 根据数据列名调整
X = df[metal_cols].copy()
# X = df[['Zn', 'Ni', 'Cu', 'Pb', 'Fe']]  # 选择重金属暴露变量
y = df['group(2对照,1病例)']

# 4.2 参数配置
n_quantiles = 4  # 分位数划分数量
n_boot = 700  # 自举次数   设置Bootstrap重抽样次数   Bootstrap通过有放回抽样估计统计量的抽样分布   科研推荐：500-1000次（需平衡计算成本与精度）
components = metal_cols  # 确保长度与mean_weights、std_weights一致
# components = X.columns.tolist()  # 混合物成分名称   X.columns → 获取DataFrame的列索引（Index对象）
np.random.seed(42)  # 固定随机种子

# 4.3 分位数转换函数
def quantile_transform(data, n_quantiles):
    # # Step1: 强制筛选数值型列并复制数据（避免修改原始DataFrame）
    # numeric_data = data.select_dtypes(include=np.number).copy()
    # # Step2: 验证数据有效性
    # if numeric_data.empty:
    #     raise ValueError("输入数据无数值型列！")
    # # Step3：分位数转换
    # return numeric_data.apply(lambda x: pd.qcut(x, n_quantiles,
    #                                     labels=False,  # 返回分位区间的整数编码（如 0,1,2,3）而非区间对象————若设置为 True 会返回类似 "(0.25, 0.5]" 的区间字符串标签
    #                                     duplicates='drop') + 1, axis=0)
    return data.apply(lambda x: pd.qcut(x, n_quantiles,
                                        labels=False,  # 返回分位区间的整数编码（如 0,1,2,3）而非区间对象————若设置为 True 会返回类似 "(0.25, 0.5]" 的区间字符串标签
                                        duplicates='drop') + 1, axis=0)
# X = df.select_dtypes(include=np.number).copy()  # 确保 X 是 DataFrame 且所有列为数值型
X_quant = quantile_transform(X, n_quantiles)  # 执行分位数转换

# 4.4 WQS核心建模函数
def wqs_model(X, y, n_boot=700):
    boot_weights = []  # 存储每次Bootstrap抽样的 特征权重（污染物贡献度）
    boot_or = []  # 存储每次抽样的 混合效应OR值（整体暴露风险）
    boot_auc = []  # 存储每次Bootstrap的AUC值

    for i in range(n_boot):
        # 自举抽样
        X_resampled, y_resampled = resample(X, y, random_state=42 + i)

        # 定义优化目标函数
        def objective(weights):
            wqs_index = np.dot(X_resampled.values, weights)
            model = LogisticRegression(penalty=None, max_iter=1000,
                                       solver='lbfgs').fit(wqs_index.reshape(-1, 1), y_resampled)
            return -model.score(wqs_index.reshape(-1, 1), y_resampled)

        # 设置约束条件（权重和为1）
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # 权重和约束
        bounds = [(0, 1) for _ in range(X.shape[1])]  # 非负约束
        init_weights = np.ones(X.shape[1]) / X.shape[1]  # 初始权重

        # 执行优化
        result = minimize(objective, init_weights,
                          method='SLSQP', bounds=bounds,
                          constraints=cons,
                          options={'maxiter': 1000})

        if result.success:  # 验证优化过程是否收敛到有效解，避免使用未收敛或错误的优化结果（如因迭代次数不足导致的失败）
            # 保存权重
            boot_weights.append(result.x)

            # 计算全数据集的WQS指数和OR值
            final_index = np.dot(X.values, result.x)
            final_model = LogisticRegression(penalty=None,
                                             max_iter=1000).fit(final_index.reshape(-1, 1), y)
            beta = final_model.coef_[0][0]
            boot_or.append(np.exp(beta))  # OR值

            # 计算当前权重下的AUC值
            fpr, tpr, _ = roc_curve(y, final_index)
            auc_value = auc(fpr, tpr)
            boot_auc.append(auc_value)  # AUC值

        # 在函数内计算置信区间
        or_ci = np.percentile(boot_or, [2.5, 97.5]) if boot_or else [np.nan, np.nan]
        auc_ci = np.percentile(boot_auc, [2.5, 97.5]) if boot_auc else [np.nan, np.nan]

    return np.array(boot_weights), np.array(boot_or), np.array(boot_auc), or_ci, auc_ci  # 返回AUC置信区间

# 4.5 调用函数时接收所有返回值   # 运行WQS模型
weights, or_values, auc_values, or_ci, auc_ci = wqs_model(X_quant, y, n_boot=700)

# 后续分析直接使用返回的置信区间
print(f"AUC 95%CI: {auc_ci[0]:.2f}-{auc_ci[1]:.2f}")

# 4.6 结果分析
# 计算统计量
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
mean_or = np.mean(or_values)

# 4.7 可视化权重分布
plt.figure(figsize=(10, 6))
plt.barh(components, mean_weights[:len(components)], xerr=std_weights[:len(components)],  # 确保mean_weights/std_weights长度与components一致）
         color='teal', alpha=0.7, capsize=5)
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel('权重系数')
plt.title('污染物权重分布（带标准差）')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# components[:len(components)] 是一个切片操作，表示从序列 components 的开头（索引 0）到其长度（即末尾）的子序列。具体来说：
# 切片的语法是 [start:end]，其中 start 默认是 0，end 默认是序列长度。
# 等价于 components[0:len(components)]，也就是整个序列本身。

# 4.8 计算综合WQS指数
wqs_index = np.dot(X_quant.values, mean_weights)

# 4.9 ROC曲线评估
fpr, tpr, _ = roc_curve(y, wqs_index)  # 真实标签 y 和预测指数 wqs_index（WQS指数）
roc_auc = auc(fpr, tpr)  # 假阳性率（FPR）、真阳性率（TPR）

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2,  # linewidth线宽
         label=f'AUC = {roc_auc:.2f} (95%CI: {auc_ci[0]:.2f}-{auc_ci[1]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # 绘制对角线，表示无判别能力的模型（AUC=0.5），用于对比实际模型性能
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('WQS模型ROC曲线')
plt.legend(loc="lower right")  # 图例显示在右下方
plt.grid(alpha=0.3)  # 添加半透明网格线(grid网格)
plt.show()

# 4.10 结果输出
print(f"混合物效应OR值: {mean_or:.2f} (95%CI: {or_ci[0]:.2f}-{or_ci[1]:.2f})")
print("\n污染物贡献权重：")
for comp, weight, std in zip(components, mean_weights, std_weights):
    print(f"• {comp}: {weight:.3f} ± {std:.3f}")  # 列出所有污染物的 权重均值 和 标准差

# 4.11 敏感性分析（可选）
print("\n权重稳定性分析：")
weight_stability = pd.DataFrame(weights, columns=components)
print(weight_stability.describe().loc[['mean', 'std', 'min', 'max']].T.round(3))

# 5. 随机森林模型
# 5.1 基础模型训练
base_rf = RandomForestClassifier(random_state=42)
base_rf.fit(X_train_scaled, y_train)

# 5.2 网格搜索调参
param_grid = {
    'n_estimators': [100, 200, 300],    # 树的数量   文献常用100-500，此处选择中等范围平衡效率
    'max_depth': [None, 10, 20],        # 树的最大深度   包含None探索数据固有复杂度
    'min_samples_split': [2, 5],       # 节点分裂最小样本数   测试模型对稀疏信号的敏感性
    'class_weight': ['balanced', None] # 处理类别不平衡   验证是否需要补偿类别不平衡
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,  # 5折交叉验证
                           scoring='roc_auc',
                           n_jobs=-1)  # 使用全部CPU核心并行计算，加速搜索过程
grid_search.fit(X_train_scaled, y_train)

# 获取最佳模型
best_rf = grid_search.best_estimator_
print(f"最佳参数组合: {grid_search.best_params_}")

# 5.3 使用最佳模型预测
y_pred = best_rf.predict(X_test_scaled)
y_proba = best_rf.predict_proba(X_test_scaled)[:, 1]  # 预测概率

# 5.4 模型评估
# 基础指标
print("\n测试集准确率: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("测试集AUC: {:.3f}".format(roc_auc_score(y_test, y_proba)))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

# 分类报告
print("\n分类报告:\n", classification_report(y_test, y_pred))

# 5.5 交叉验证评估稳定性
cv_scores = cross_val_score(best_rf,  # 交叉 验证
                           X_train_scaled,
                           y_train,
                           cv=5,
                           scoring='roc_auc')  # 使用AUC（ROC曲线下面积）作为评估指标
print("交叉验证AUC: {:.3f} (±{:.3f})".format(cv_scores.mean(), cv_scores.std()))

# 5.6 ROC曲线绘制
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC曲线 (AUC = %0.2f)' % roc_auc)  # plt.plot(x, y, 样式参数)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])  # x轴范围固定为0到1   FPR的理论范围为[0,1]，固定显示避免空白。
plt.ylim([0.0, 1.05])  # y轴范围扩展至1.05   TPR通常≤1，但扩展5%空间防止图例或标签被截断。
plt.xlabel('假阳性率')  # x轴含义(标签)
plt.ylabel('真阳性率')  # y轴含义(标签)
plt.title('随机森林ROC曲线')
plt.legend(loc="lower right")  # 图例位置在右下方
plt.show()

# 5.7 特征重要性分析
importances = best_rf.feature_importances_  # 提取模型学习到的特征重要性得分（数值化评估特征贡献）
indices = np.argsort(importances)[::-1]  # 生成按重要性 降序排列 的索引，用于后续可视化或分析时的特征排序

# 垂直条形图
plt.figure(figsize=(10, 6))  # 宽 高   加宽加高画布适应更多特征，避免标签重叠
plt.title("随机森林特征重要性排序", fontsize=14, pad=20)
plt.xlabel("重要性得分", fontsize=12, labelpad=10)
plt.ylabel("特征名称", fontsize=12, labelpad=10)
plt.bar(range(X_train.shape[1]), importances[indices],
        color="skyblue", align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices],
           rotation=45, ha='right')  # 45度倾斜+右对齐防止重叠
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()  # 自动调整布局
plt.show()



# 6. XGboost模型

# 6.1 数据格式转换
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# 6.2 参数设置
params = {
    'objective': 'binary:logistic',  # 二分类任务
    'eval_metric': 'auc',           # 评估指标
    'eta': 0.1,                     # 学习率
    'max_depth': 5,                 # 树的最大深度
    'subsample': 0.8,               # 样本采样比例
    'colsample_bytree': 0.8,        # 特征采样比例
    'seed': 42,                     # 随机种子
    'scale_pos_weight': np.sqrt(len(y_train[y_train==0])/len(y_train[y_train==1]))  # 处理类别不平衡   改进公式（平方根法）
}

# 6.3 交叉验证调参
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=500,           # 最大迭代次数
    nfold=5,                       # 5折交叉验证
    metrics=['auc'],
    early_stopping_rounds=20,      # 早停轮数
    verbose_eval=50                # 每50轮显示进度
)

# 获取最优迭代次数
best_rounds = cv_results.shape[0]

# 6.4 模型训练
model_xgb = xgb.train(
    params,
    dtrain,
    num_boost_round=best_rounds,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    verbose_eval=50
)

# 6.5 模型预测
y_pred_prob = model_xgb.predict(dtest)  # 预测概率值
y_pred = (y_pred_prob > 0.5).astype(int)  # 概率转类别标签

# 6.6 模型评估
print("\n=== 模型评估结果 ===")
# 基础指标
print("测试集准确率: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("测试集AUC: {:.3f}".format(roc_auc_score(y_test, y_pred_prob)))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

# 分类报告
print("\n分类报告:\n", classification_report(y_test, y_pred))

# 6.7 ROC曲线绘制
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC曲线 (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('XGBoost模型ROC曲线')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# 6.8 特征重要性分析
plt.figure(figsize=(10,6))
xgb.plot_importance(model_xgb,
                   # importance_type='weight',  # 按特征使用次数计算重要性
                   importance_type='gain',  # 科研应用建议优先使用(直接反映污染物浓度变化对癌症风险的贡献度)
                   max_num_features=15,        # 显示top15特征
                   height=0.8,
                   color='#2874A6')
plt.title('XGBoost特征重要性排名', fontsize=14)
plt.xlabel('F分数', fontsize=12)
plt.ylabel('特征名称', fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.show()

# 6.9 保存模型
model_xgb.save_model('xgb_model.json')  # 保存为JSON格式


# 7. 支持向量机(SVM)模型(二分类SVM)
# 7.1 基础(未经调参)SVM模型   base_svm是个模型
base_svm = SVC(probability=True, random_state=42)  # 需要概率估计来进行后续的AUC评估
base_svm.fit(X_train_scaled, y_train)  # # 使用标准化后的训练数据训练

# 7.2 网格搜索调参
param_grid = {
    'C': [0.1, 1, 10, 100],  # 正则化参数 作用：控制模型的复杂度和容错能力
    'gamma': ['scale', 'auto', 0.01, 0.1],  # 核函数系数 作用：控制单个样本对决策边界的影响范围（仅对RBF核有效）
    'kernel': ['rbf', 'linear'],  # 核函数类型 作用：决定数据映射到高维空间的方式
    'class_weight': [None, 'balanced']  # 处理类别不平衡
}

grid_search = GridSearchCV(SVC(probability=True, random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='roc_auc',
                           n_jobs=-1,
                           verbose=1)  # 显示调参过程
grid_search.fit(X_train_scaled, y_train)

# 7.3 获取最佳模型
best_svm = grid_search.best_estimator_  # 获取最佳模型实例
print(f"\n最佳参数组合: {grid_search.best_params_}")  # 打印经调优后的模型的最佳参数组合

# 7.4 模型预测
y_pred = best_svm.predict(X_test_scaled)
y_proba = best_svm.predict_proba(X_test_scaled)[:, 1]
# 同5. 随机森林模型

# 7.5 模型评估
print("\n=== SVM模型评估结果 ===")
print("测试集准确率: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("测试集AUC: {:.3f}".format(roc_auc_score(y_test, y_proba)))
# 同5. 随机森林模型

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('SVM混淆矩阵')
plt.show()
# 同5. 随机森林模型

# 分类报告
print("\n分类报告:\n", classification_report(y_test, y_pred))
# 同5. 随机森林模型

# 7.6 交叉验证评估
cv_scores = cross_val_score(best_svm,
                            X_train_scaled,
                            y_train,
                            cv=5,
                            scoring='roc_auc')
print("交叉验证AUC: {:.3f} (±{:.3f})".format(cv_scores.mean(), cv_scores.std()))
# 同5. 随机森林模型

# 7.7 ROC曲线绘制
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC曲线 (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('SVM模型ROC曲线')
plt.legend(loc="lower right")
plt.grid(alpha=0.3) # 比 5.随机森林模型多一行对透明度的设置
plt.show()

# 7.8 特征分析（线性核时可用）
if best_svm.kernel == 'linear':
    coef_df = pd.DataFrame({
        'Feature': X.columns,  # 特征名称列
        'Coefficient': best_svm.coef_[0]    # SVM系数列
    }).sort_values('Coefficient', ascending=False)  # 按系数绝对值降序排序

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')  # 数据可视化中用于指定颜色方案的参数
    plt.title('SVM特征系数（线性核）')
    plt.xlabel('系数值')
    plt.ylabel('特征名称')
    plt.tight_layout()
    plt.show()
else:
    print("\n提示：使用非线性核时无法直接显示特征重要性，建议使用置换重要性分析")