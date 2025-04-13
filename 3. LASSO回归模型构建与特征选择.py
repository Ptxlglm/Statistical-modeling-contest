# 3. LASSO回归模型构建与特征选择
# 3.1 初始化LASSO回归模型（自动交叉验证选择最优alpha）
lasso = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10],  # 正则化强度候选值
                cv=5,  # 5折交叉验证
                max_iter=10000,  # 最大迭代次数，确保收敛
                random_state=42,  # 随机种子，便于实验复现
                n_jobs=-1)  # 使用全部CPU核心，并行计算加速

# 3.2 训练模型
lasso.fit(X_train_scaled, y_train)
# X_train_scaled：经过标准化处理的训练集特征数据（如归一化处理后的数值）。
# y_train：训练集的标签数据（预测目标）。
# fit()：训练方法，让模型从数据中学习特征与标签的关系。

# 3.3 获取特征系数
feature_coef = pd.DataFrame({  # 创建一个excel表
    'Feature': X.columns,  # X.columns：表示数据集 X 的所有特征名称（如 ["年龄", "收入", "性别"]）
    'Coefficient': lasso.coef_,  # LASSO回归模型的特征系数值（包含正负），反映每个特征对目标变量（如结直肠癌发病风险）的影响方向和强度
    'Absolute_Coef': abs(lasso.coef_)  # 系数的绝对值（衡量重要性）
}).sort_values('Absolute_Coef', ascending=False)  # 按绝对值降序排序
# Coefficient(系数)————lasso.coef_
# 正系数：特征与发病风险正相关
# 示例：Blood_Pb（血铅）系数+0.83 → 血铅每升高1单位，风险增加0.83单位
# 负系数：特征与发病风险负相关
# 示例：BMI系数-0.15 → BMI每增加1单位，风险降低0.15单位
# 零系数：该特征被模型剔除

# 3.4 筛选重要特征（系数非零的特征）
selected_features = feature_coef[feature_coef['Coefficient'] != 0]['Feature'].tolist()
print("\nLASSO选择的重要特征（非零系数）:")
print(selected_features)
# feature_coef['Coefficient'] != 0筛选出系数非零的特征行
# 数学原理：LASSO的L1正则化将弱相关特征的系数β压缩至0
#
# feature_coef[ ... ]应用布尔索引，提取所有非零系数对应的数据行
# ['Feature']从筛选结果中提取特征名称列，准备输出为列表
#
# .tolist()将Pandas Series转换为Python列表，便于后续建模使用

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

# coef_plot = feature_coef[feature_coef['Coefficient'] != 0].sort_values('Coefficient')
# plt.barh(coef_plot['Feature'], coef_plot['Coefficient'], color='steelblue')
# 绘制系数条形图

# plt.axvline(0, color='black', linestyle='--', linewidth=1)
# 在图表中 垂直方向（x轴方向） 添加一条参考线，位置在 x=0 处。
# 作用
    # 分割正负效应：将特征的正向（右侧）和负向（左侧）影响直观分开
    # 视觉基线：帮助观察特征系数偏离零的程度（离基线越远，作用越强）
    # 科研图表规范：符合学术期刊对统计图表需标注基准线的要求（如《Nature》图表指南）

# 预测效果散点图(第二个子图)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolor='k')  # 横坐标数据：测试集真实值   纵坐标数据：模型预测值   点透明度alpha最好在0.3~0.8之间	避免重叠点掩盖分布密度，0.6 平衡可见性与重叠显示
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('真实值 vs 预测值')
plt.xlabel('真实值')
plt.ylabel('预测值')

# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
# 在散点图上绘制一条绘制一条从 真实值(min_value, min_value) 到 真实值(max_value, max_value) 的直线，即y = x的理想预测线
# X轴：真实值最小值 → 最大值          [min(y_test), max(y_test)],
# Y轴：与X轴完全相同                 [min(y_test), max(y_test)],
# 红色虚线                          'r--'
# 红色虚线：完美预测时所有点应落在此线上
# 实际点位置：
# 点在线上方 → 模型高估（预测值 > 真实值）
# 点在线下方 → 模型低估（预测值 < 真实值）

plt.tight_layout()  # 自动调整子图间距，防止标签重叠
plt.show()

# 3.7 保存重要特征结果（供后续建模使用）
selected_features_df = pd.DataFrame({'Selected_Features': selected_features})  # 创建特征列表的DataFrame
selected_features_df.to_csv('selected_features_lasso.csv', index=False)  # 保存到CSV文件（无行索引）