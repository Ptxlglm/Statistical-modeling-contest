# 4. WQS回归模型构建与混合物效应分析
# 数据准备
# 根据前面预处理后的数据
X = df[['Zn', 'Ni', 'Cu', 'Pb', 'Fe', 'Cd']]  # 选择重金属暴露变量
y = df['group(2对照,1病例)']

# 参数配置
n_quantiles = 4  # 分位数划分数量
n_boot = 700  # 自举次数   设置Bootstrap重抽样次数   Bootstrap通过有放回抽样估计统计量的抽样分布   科研推荐：500-1000次（需平衡计算成本与精度）
components = X.columns.tolist()  # 混合物成分名称   X.columns → 获取DataFrame的列索引（Index对象）
np.random.seed(42)  # 固定随机种子

# 分位数转换函数
def quantile_transform(data, n_quantiles):
    # Step1: 强制筛选数值型列并复制数据（避免修改原始DataFrame）
    numeric_data = data.select_dtypes(include=np.number).copy()
    # Step2: 验证数据有效性
    if numeric_data.empty:
        raise ValueError("输入数据无数值型列！")
    # Step3：分位数转换
    return numeric_data.apply(lambda x: pd.qcut(x, n_quantiles,
                                        labels=False,  # 返回分位区间的整数编码（如 0,1,2,3）而非区间对象————若设置为 True 会返回类似 "(0.25, 0.5]" 的区间字符串标签
                                        duplicates='drop') + 1, axis=0)
# 若省略 .copy()，直接修改 numeric_data 可能导致原始 data 被意外修改（Pandas的视图机制）
# 在numeric_data中apply(添加)分位等级编码这一列
# Lambda 函数的语法(匿名函数，单行、无函数名、可传参数给高阶函数，简洁、可读性降低、不可多行、难调试)
    # lambda 参数列表: 表达式
        # eg.
        # add = lambda x, y: x + y
        # print(add(2, 3))  # 输出 5
# pd.cut()方法将连续变量按分位数区间离散化，生成分位等级编码。
# duplicates='drop'
# 自动删除重复的分位边界，避免因数据重复导致的分位冲突
# 适用场景：当数据中存在大量重复值时（如检测限以下的值全为0），分位边界可能重复
# +1
# 将分位编码从 0-based（0,1,2,3）转换为 1-based（1,2,3,4）
# axis=0
# 对 DataFrame 的每一列独立应用分位数转换
# 注意：
# axis=0 沿行方向操作（即 按列处理）
# axis=1 沿列方向操作（即 按行处理）

# data = df.select_dtypes(include=np.number).columns
# 注意：.columns表示转换为列名列表&&函数本身缺乏防御性

X = df.select_dtypes(include=np.number).copy()  # 确保 X 是 DataFrame 且所有列为数值型
X_quant = quantile_transform(X, n_quantiles)  # 执行分位数转换
# X是原始的数值型DataFrame，X_quant是转换后的分位数等级DataFrame(每个数值被替换为对应的 分位等级（1到n_quantiles）)
# .copy()
# 作用：创建数据的独立副本，避免后续操作修改原始数据
# 必要性：若省略，对 X 的修改可能影响原始 df（Pandas 默认返回视图而非副本）

# WQS核心建模函数
def wqs_model(X, y, n_boot=700):
    boot_weights = []  # 存储每次Bootstrap抽样的 特征权重（污染物贡献度）
    boot_or = []  # 存储每次抽样的 混合效应OR值（整体暴露风险）
    boot_auc = []  # 存储每次Bootstrap的AUC值

    for i in range(n_boot):
        # 自举抽样
        X_resampled, y_resampled = resample(X, y, random_state=42 + i)
        # 有放回地抽取与原始数据相同大小的样本
        # random_state = 42 + i
        # 确保每次抽样结果可复现且独立
        # 通过多次重抽样构建 加权分位数和（WQS）指数，评估污染物混合效应的稳健性

        # 定义优化目标函数
        def objective(weights):
            wqs_index = np.dot(X_resampled.values, weights)
            model = LogisticRegression(penalty=None, max_iter=1000,
                                       solver='lbfgs').fit(wqs_index.reshape(-1, 1), y_resampled)
            return -model.score(wqs_index.reshape(-1, 1), y_resampled)
        # weights ：
            # 类型：一维数组（长度 = 特征数）
            # 含义：待优化的污染物权重，满足非负且和为1的约束
            # 示例：[0.3, 0.5, 0.2]表示3个污染物的混合效应权重
        # wqs_index = np.dot(X_resampled.values, weights)
            # 计算混合暴露指数（WQS指数）
            # np.dot: 计算 加权分位数和（WQS指数）
            # 公式：
            # WQS = X1×w1+X2×w2+⋯+Xk×wk
            # 输出：一维数组（每个样本的混合暴露指数）
            # X_resampled：Bootstrap抽样后的特征矩阵（形状：(样本数, 特征数)）
            # .values
            # 将 DataFrame 转换为 NumPy 数组（形状：样本数 × 污染物数）
            # weights：当前迭代的权重向量
        # model = LogisticRegression(penalty=None, max_iter=1000)
            # 目的：训练逻辑回归模型，量化WQS指数与疾病风险的关联
            # penalty = None：禁用正则化（避免干扰权重优化）
            # max_iter = 1000：确保复杂数据下的收敛性
            # solver = 'lbfgs'：适用于小到中型数据集的优化算法
        # -model.score(wqs_index.reshape(-1, 1), y_resampled)
            # 最大化模型准确率 → 等价于最小化负准确率
            # model.score : 计算模型在训练数据上的准确率（取值范围[0, 1]）
            # wqs_index.reshape(-1, 1)：
            # reshape(-1,1):将WQS指数转为二维数组（模型输入格式）
            # y_resampled：Bootstrap抽样后的标签（病例 / 对照）
            # 输出：负准确率：将最大化准确率问题转换为最小化问题（优化器寻找最小值即等价于寻找最高准确率）（优化器默认寻找最小值）

        # 优化目标：寻找使逻辑回归模型准确率最高的权重组合
        # 输出驱动：权重优化结果反映污染物对疾病风险的联合效应强度

        # 设置约束条件（权重和为1）
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # 权重和约束
        bounds = [(0, 1) for _ in range(X.shape[1])]  # 非负约束
        init_weights = np.ones(X.shape[1]) / X.shape[1]  # 初始权重
        # cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            # cons————constraint约束条件
            # 约束类型'type': 'eq' → 等式约束
            # 约束函数'fun'：lambda w: np.sum(w) - 1 → 计算权重和与1的差值（要求
            # np.sum(w) = 1）
            # 作用：确保所有污染物的权重之和为1（即权重表示比例贡献）
        # bounds = [(0, 1) for _ in range(X.shape[1])]  # 非负约束
            # 边界条件bounds
            # (0, 1) → 每个权重wi的取值范围为[0, 1]
            # X.shape[1] → 污染物数量（即权重数量）
            # 作用：定义每个权重的 取值范围，保证每个污染物的权重非负且不超过1（满足 WQS 模型假设）
        # init_weights = np.ones(X.shape[1]) / X.shape[1]
            # 初始权重init_weights
            # np.ones(X.shape[1]) → 生成全1数组（长度 = 污染物数量）
            # / X.shape[1] → 归一化为均等权重（如3个污染物时初始权重为[0.33, 0.33, 0.33]）
            # 作用：设置优化算法的 初始猜测值，为优化器提供一个合理的起点，加速收敛并减少局部最优风险

        # 通过约束条件和边界条件，确保优化后的权重满足 方向一致性假设（所有污染物效应方向相同）和 可解释性（权重总和为1，表示相对贡献比例）

        # 执行优化
        result = minimize(objective, init_weights,
                          method='SLSQP', bounds=bounds,
                          constraints=cons,
                          options={'maxiter': 1000})
        #  ** objective **: 要最小化的目标函数（此处为返回负准确率的函数）
        #  ** init_weights **: 初始权重猜测（均分权重，如3个污染物则为[0.33, 0.33, 0.33]）
        #  ** method = 'SLSQP': 序列最小二乘规划算法，专为处理 约束优化问题 设计
        #  ** bounds = bounds **: 权重的取值范围约束（每个权重 ∈ [0, 1]）
        #  ** constraints = cons **: 等式约束（权重总和必须为1）
        #  ** options = {'maxiter': 1000} **: 最大迭代次数（避免未收敛时提前终止）

        # 这段代码使用scipy.optimize.minimize方法求解 带约束的最优化问题，
        # 其核心目标是找到一组污染物权重，使得 加权分位数和（WQS）指数对疾病风险的预测准确率最高

        # 通过优化算法寻找满足约束条件的权重组合，使得逻辑回归模型的预测准确率最高

        # 优化目标
        # 寻找满足以下条件的污染物权重组合
        # w = [w1, w2, ..., wk]：
        # 1、约束条件：
        # ∑(i = 1,k) wi = 1
        # wi∈[0, 1]
        # 2、优化目标：最大化逻辑回归模型的预测准确率（等价于最小化负准确率 −Accuracy）。

        if result.success:  # 验证优化过程是否收敛到有效解，避免使用未收敛或错误的优化结果（如因迭代次数不足导致的失败）
            # 保存权重
            boot_weights.append(result.x)

            # # 计算全数据集的WQS指数和OR值
            final_index = np.dot(X.values, result.x)
            final_model = LogisticRegression(penalty=None,
                                             max_iter=1000).fit(final_index.reshape(-1, 1), y)
            beta = final_model.coef_[0][0]
            boot_or.append(np.exp(beta))  # OR值

            # 计算当前权重下的AUC值
            fpr, tpr, _ = roc_curve(y, final_index)
            auc_value = auc(fpr, tpr)
            boot_auc.append(auc_value)  # AUC值

            # result.x：优化后的权重向量
            # boot_weights：列表，存储所有Bootstrap抽样的权重结果，用于后续计算权重均值和置信区间

            # X.values：将DataFrame转换为NumPy数组
            # np.dot：矩阵乘法，计算加权分位数和（WQS指数）

            # final_model = LogisticRegression(penalty=None,
            #                                  max_iter=1000).fit(final_index.reshape(-1, 1), y)
            # 拟合逻辑回归模型
            # ** penalty = None **：禁用正则化，避免偏差引入权重估计
            # ** max_iter = 1000 **：确保模型收敛
            # ** final_index.reshape(-1, 1) **：将一维WQS指数转为二维数组（形状：(样本数, 1)），满足scikit - learn输入格式
            # ** y **：原始目标变量（病例 / 对照标签）

            # coef_：逻辑回归系数数组
            # beta：WQS指数每增加1单位对应的 对数风险比

            # np.exp(beta)：将对数风险比转换为 比值比
            # ** boot_or **：列表，存储所有Bootstrap抽样的OR值，用于计算总体OR及其置信区间

        # 在函数内计算置信区间
        or_ci = np.percentile(boot_or, [2.5, 97.5])
        auc_ci = np.percentile(boot_auc, [2.5, 97.5])
        # 输出：or_ci（置信区间上下限，如 [1.2, 2.0]）
        # 作用：量化混合物效应 OR 值的统计显著性（若区间不包含1，则效应显著）。

    return np.array(boot_weights), np.array(boot_or), np.array(boot_auc), or_ci, auc_ci  # 返回AUC置信区间

# 调用函数时接收所有返回值   # 运行WQS模型
weights, or_values, auc_values, or_ci, auc_ci = wqs_model(X_quant, y, n_boot=700)
# 将列表 boot_weights（Bootstrap权重集合）和 boot_or（Bootstrap OR值集合）
# 转换为 NumPy数组，返回形式为 (权重矩阵, OR值数组)

# 后续分析直接使用返回的置信区间
print(f"AUC 95%CI: {auc_ci[0]:.2f}-{auc_ci[1]:.2f}")

# 结果分析
# 计算统计量
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
mean_or = np.mean(or_values)
# 使用 np.percentile 计算 or_values 数组的 2.5% 和 97.5% 分位数，
# 得到 95% 置信区间（95% Confidence Interval, CI）
# 输入：or_values（Bootstrap 抽样后的 OR 值数组，形状 (n_boot,)）

# 可视化权重分布
plt.figure(figsize=(10, 6))
plt.barh(components, mean_weights, xerr=std_weights,
         color='teal', alpha=0.7, capsize=5)
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel('权重系数')
plt.title('污染物权重分布（带标准差）')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# **components**：污染物名称列表（Y轴标签）
# **mean_weights**：污染物权重均值（条形的长度）
# **xerr=std_weights**：误差线（标准差，反映权重波动性）
# **capsize=5**：误差线端帽长度（像素）

# plt.gca().invert_yaxis()
# 功能：反转Y轴，使条形图从上到下显示污染物（默认从下到上）
# 意义：符合阅读习惯（重要污染物靠上显示）

# plt.grid(axis='x', linestyle='--', alpha=0.7)
# 功能：添加 X轴虚线网格
# **axis='x'**：仅在X轴方向显示网格
# **linestyle='--'**：虚线样式
# **alpha=0.7**：网格透明度（避免喧宾夺主）

# 整体意义
# 生成一张 污染物权重分布图，直观展示：
#
# 各污染物的 平均贡献权重（条形长度）
# 权重的 波动范围（误差线，反映Bootstrap结果的稳定性）
# 通过颜色、布局和网格优化可读性，便于快速识别关键污染物。

# 计算综合WQS指数
wqs_index = np.dot(X_quant.values, mean_weights)

# ROC曲线评估
fpr, tpr, _ = roc_curve(y, wqs_index)  # 真实标签 y 和预测指数 wqs_index（WQS指数）
roc_auc = auc(fpr, tpr)  # 假阳性率（FPR）、真阳性率（TPR）
# auc：计算ROC曲线下面积（AUC），量化模型区分病例与对照的能力（0.5~1，越大越好）

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

# 整体意义
# 通过ROC曲线和AUC值 评估WQS模型对病例/对照的区分能力：
# AUC > 0.7：模型具有较好判别能力
# AUC接近1：模型几乎完美区分病例与对照

# 结果输出
print(f"混合物效应OR值: {mean_or:.2f} (95%CI: {or_ci[0]:.2f}-{or_ci[1]:.2f})")
print("\n污染物贡献权重：")
for comp, weight, std in zip(components, mean_weights, std_weights):
    print(f"• {comp}: {weight:.3f} ± {std:.3f}")  # 列出所有污染物的 权重均值 和 标准差

# print(f"混合物效应OR值: {mean_or:.2f} (95%CI: {or_ci[0]:.2f}-{or_ci[1]:.2f})")
# 输出加权分位数和（WQS）指数对应的 混合效应比值比（OR） 及其 95%置信区间

# 敏感性分析（可选）
print("\n权重稳定性分析：")
weight_stability = pd.DataFrame(weights, columns=components)
print(weight_stability.describe().loc[['mean', 'std', 'min', 'max']].T.round(3))
# .describe()：计算均值、标准差、最小值、最大值（loc[['mean', 'std', 'min', 'max']]）等数值列统计量
# 输出
#            mean   std   min   max
# Pb        0.450 0.080 0.310 0.590
# Cd        0.350 0.070 0.220 0.480
