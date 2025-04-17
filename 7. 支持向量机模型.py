# 7. 支持向量机(SVM)模型(二分类SVM)
# 目标变量只有0（对照组）和1（病例组）两个类别
# 使用SVC类默认实现二元分类
# 7.1 基础(未经调参)SVM模型   base_svm是个模型
base_svm = SVC(probability=True, random_state=42)  # 需要概率估计来进行后续的AUC评估
base_svm.fit(X_train_scaled, y_train)  # # 使用标准化后的训练数据训练
# SVC是Scikit-learn中的支持向量分类器，用于分类任务
# probability=True,
# 作用：启用概率估计功能
# 原理：通过Platt缩放方法将SVM的决策函数值转换为类概率（需要额外计算）
# 必要性：为后续计算ROC曲线和AUC值提供概率预测
# 若设置为False：将无法使用.predict_proba()方法获取概率值

# X_train_scaled：经过标准化处理的训练集特征数据（如归一化处理后的数值）。
# y_train：训练集的标签数据（预测目标）。
# fit()：训练方法，让模型从数据中学习特征与标签的关系。

# base_svm.fit(X_train_scaled, y_train)  使用默认参数训练基础模型
# 阶段一：基础模型训练

# 7.2 网格搜索调参
param_grid = {
    'C': [0.1, 1, 10, 100],  # 正则化参数 作用：控制模型的复杂度和容错能力
    'gamma': ['scale', 'auto', 0.01, 0.1],  # 核函数系数 作用：控制单个样本对决策边界的影响范围（仅对RBF核有效）
    'kernel': ['rbf', 'linear'],  # 核函数类型 作用：决定数据映射到高维空间的方式
    'class_weight': [None, 'balanced']  # 处理类别不平衡
}
# 正则化参数C 工作原理：
# C值越小 → 正则化强度越大 → 允许更多误分类 → 模型简单（防止过拟合）
# C值越大 → 正则化强度越小 → 追求更高分类精度 → 模型复杂（可能过拟合

# 核函数系数
# 取值范围：['scale', 'auto', 0.01, 0.1]
# 具体含义：
# 'scale'：默认值，1/(n_features * X.var())
# 'auto'：1/n_features
# 数值型：直接指定gamma值
# 影响规律：
# gamma值大 → 样本影响范围小 → 决策边界曲折（容易过拟合）
# gamma值小 → 样本影响范围大 → 决策边界平滑（可能欠拟合）
# 典型场景：
# 0.01：适用于特征间相关性较强的数据
# 0.1：适用于特征维度较低的情况

# 'kernel': ['rbf', 'linear']
# 选项：['rbf', 'linear']
# 核函数对比：
# 核类型	特点	适用场景
# rbf	非线性核，通过高斯函数映射	特征间存在复杂非线性关系
# linear	线性核，直接进行线性划分	高维数据/特征间线性可分
# 选择建议：
# 当数据维度 > 样本量时优先尝试linear
# 当特征交互复杂时使用rbf

# 'class_weight': [None, 'balanced']
# 选项：[None, 'balanced']
# 工作方式：
# None：默认等权重（所有类别权重=1）
# 'balanced'：自动计算权重，权重=总样本数/(类别数*各类样本数)
# 示例：
# 若正负样本比例为1:99
# 'balanced'会给正样本分配99倍于负样本的权重
# 使用建议：
# 当类别比例 > 1:5时建议开启
# 医学研究中常对罕见病案例启用

grid_search = GridSearchCV(SVC(probability=True, random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='roc_auc',
                           n_jobs=-1,
                           verbose=1)  # 显示调参过程
grid_search.fit(X_train_scaled, y_train)
# 在Scikit-learn的GridSearchCV中，verbose参数控制着网格搜索过程中信息输出的详细程度
# 参数值	输出级别	适用场景
# 0	静默模式（不输出）	生产环境/不需要监控时
# 1	精简进度	交互式环境/需要监控进度时
# ≥2	详细调试信息	深度调试/查看每个参数组合结果

# grid_search.fit(X_train_scaled, y_train)  在同一训练集上寻找最优参数组合
# 阶段二：参数调优

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
# 选用线性核：
# 识别对结直肠癌发病风险有独立影响的重金属
# 量化每种重金属的效应方向（促进/抑制）和相对强度
# WQS已经进行了混合效应分析
if best_svm.kernel == 'linear':
    coef_df = pd.DataFrame({
        'Feature': X.columns,  # 特征名称列
        'Coefficient': best_svm.coef_[0]    # SVM系数列
    }).sort_values('Coefficient', ascending=False)  # 按系数绝对值降序排序
    # best_svm.coef_[0]
    # 仅当SVM使用线性核（kernel = 'linear'）时有效
    # 表示各特征在决策函数中的权重系数
    # 系数的绝对值大小反映特征重要性

    # ascending = False
    # 实现从大到小排序
    # 正系数 → 特征与目标变量正相关
    # 负系数 → 特征与目标变量负相关

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')  # 数据可视化中用于指定颜色方案的参数
    plt.title('SVM特征系数（线性核）')
    plt.xlabel('系数值')
    plt.ylabel('特征名称')
    plt.tight_layout()
    plt.show()
else:
    print("\n提示：使用非线性核时无法直接显示特征重要性，建议使用置换重要性分析")

#               线性核 (kernel='linear')	               非线性核 (如kernel='rbf')
# 决策边界	      超平面（直线/平面）	                   复杂曲面（如环形、波浪形）
# 数学表达	      f(x)=w<T>x+b	                       f(x)=∑αiyiK(xi,x)+b
# 重金属数据适用性 适合重金属浓度与癌症风险呈线性剂量效应关系     适合存在协同/拮抗效应（如Cu与Zn的交互作用）
# 计算复杂度      O(n_features)	                       O(n^2_samples2 * n_features)
# 特征解释性	  可通过系数直接解释特征影响方向	               需使用SHAP值等事后解释方法
# 典型场景	      初步探索性分析、需要医学解释性的研究	       复杂暴露模式分析、多污染物交互效应研究
#
# 医学研究中的选择建议：
#     当研究某单一重金属的主效应时优先使用线性核
#     当关注多种污染物的联合暴露效应时使用RBF核
#     若最终选择非线性核，需补充敏感性分析（如改变gamma值）

# viridis特性：
# 渐进色系：从深紫色（低值）→ 蓝绿色（中值）→ 亮黄色（高值）
# 优点：
# 符合人类视觉感知的线性变化
# 对色盲人群友好
# 打印友好（黑白打印时仍能区分）
# 以下是 `sns.barplot()`、`plt.bar()` 和 `plt.barh()` 的异同详解，帮助您根据需求选择合适的方法：
#
# ---
#
# ### **1. 基本归属**
# | 方法                | 所属库      | 面向场景       |
# |---------------------|-------------|----------------|
# | `sns.barplot()`     | Seaborn     | 统计可视化     |
# | `plt.bar()`         | Matplotlib  | 基础垂直条形图 |
# | `plt.barh()`        | Matplotlib  | 基础水平条形图 |
#
# ---
#
# ### **2. 核心差异**
# #### **(1) 输入数据格式**
# ```python
# # Seaborn (长格式数据)
# sns.barplot(x="category", y="value", hue="group", data=df)
#
# # Matplotlib (需明确坐标)
# plt.bar(x_pos, heights)          # 垂直
# plt.barh(y_pos, widths)          # 水平
# ```
#
# #### **(2) 统计功能**
# | 功能                | `sns.barplot()` | `plt.bar()`/`plt.barh()` |
# |---------------------|-----------------|--------------------------|
# | 自动计算均值         | ✔️              | ❌（需手动计算）         |
# | 显示置信区间         | ✔️ (`ci`参数)   | ❌                       |
# | 分组对比 (`hue`)     | ✔️              | ❌（需手动定位）         |
#
# #### **(3) 方向控制**
# ```python
# # Seaborn 方向控制
# sns.barplot(x="value", y="category", orient="h")  # 水平
#
# # Matplotlib 方向控制
# plt.bar(...)   # 固定垂直
# plt.barh(...)  # 固定水平
# ```
#
# ---
#
# ### **3. 默认样式对比**
# | 特性                | Seaborn                     | Matplotlib                 |
# |---------------------|-----------------------------|----------------------------|
# | 颜色搭配            | 多色系自动分组              | 单一颜色（需手动设置）    |
# | 坐标轴标签          | 自动从DataFrame获取         | 需手动添加`xlabel/ylabel`  |
# | 网格线              | 默认开启浅色网格            | 默认无网格                |
# | 误差线              | 默认显示（可关闭）          | 需手动添加`errorbar`       |
#
# ---
#
# ### **4. 代码复杂度示例**
# #### **(1) 分组条形图**
# ```python
# # Seaborn 实现
# sns.barplot(x="year", y="sales", hue="product", data=df)
#
# # Matplotlib 实现
# years = df['year'].unique()
# products = df['product'].unique()
# bar_width = 0.35
#
# for i, product in enumerate(products):
#     offsets = [x + i*bar_width for x in range(len(years))]
#     values = df[df['product']==product]['sales'].values
#     plt.bar(offsets, values, width=bar_width, label=product)
# ```
#
# #### **(2) 带误差线的条形图**
# ```python
# # Seaborn 自动计算
# sns.barplot(x="group", y="score", data=df, ci=95)
#
# # Matplotlib 手动实现
# means = df.groupby('group')['score'].mean()
# stds = df.groupby('group')['score'].std()
# x_pos = range(len(means))
#
# plt.bar(x_pos, means, yerr=stds, capsize=5)
# ```
#
# ---
#
# ### **5. 使用场景建议**
# | 场景                          | 推荐方法          | 理由                                                                 |
# |-------------------------------|-------------------|----------------------------------------------------------------------|
# | 快速探索数据分布              | `sns.barplot()`   | 自动统计计算+美观呈现                                               |
# | 论文图表制作                  | `sns.barplot()`   | 内置误差线符合学术规范                                              |
# | 动态交互可视化                | `plt.bar()`       | 与GUI框架（如Tkinter）集成更方便                                    |
# | 超多类别展示（>15个）         | `plt.barh()`      | 水平布局避免标签重叠                                                |
# | 高度定制化图表（如特殊动画）  | `plt.bar()`       | 底层API控制更灵活                                                  |
#
# ---
#
# ### **6. 性能注意事项**
# - **大数据集**（>10万条数据）：
#   - 避免使用 `sns.barplot()`（计算统计量慢）
#   - 改用 `plt.hist()` 或 `plt.bar()` 直接绘制预处理结果
# - **实时更新图表**：
#   - Matplotlib 的 `plt.bar()` 配合 `animation` 模块更高效
# - **内存限制**：
#   - `sns.barplot()` 会缓存更多中间数据（如置信区间计算）
#
# 掌握这些差异后，您可以根据数据特征和展示需求选择最合适的可视化工具。