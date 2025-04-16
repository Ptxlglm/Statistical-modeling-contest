# 5. 随机森林模型
# 5.1 基础模型训练
base_rf = RandomForestClassifier(random_state=42)
base_rf.fit(X_train_scaled, y_train)
# X_train_scaled：经过标准化处理的训练集特征数据（如归一化处理后的数值）。
# y_train：训练集的标签数据（预测目标）。
# fit()：训练方法，让模型从数据中学习特征与标签的关系。

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
# n_estimators
# 含义：随机森林中决策树的数量。
# 候选值：[100, 200, 300]
# 作用：
# 树越多，模型越稳定，但计算成本越高。
# 值过小（如100）可能导致欠拟合，值过大（如300）可能提升性能但边际效益递减。
# 科研意义：权衡模型性能与计算效率。

# max_depth
# 含义：单棵决策树的最大深度。
# 候选值：[None, 10, 20]
# None：不限制深度，树会生长到所有叶子节点纯净或达到min_samples_split限制。
# 10/20：限制树深，防止过拟合。
# 作用：
# 深度越大，模型越复杂，可能过拟合；深度过小可能欠拟合。

# min_samples_split
# 含义：节点分裂所需的最小样本数。
# 候选值：[2, 5]
# 作用：
# 值越小（如2），树越深、越复杂，可能过拟合。
# 值越大（如5），树生长更保守，防止噪声学习。
# 科研意义：探索数据中局部模式的有效性。

# class_weight
# 含义：处理类别不平衡的权重设置。
# 候选值：
# 'balanced'：自动按类别频率调整权重（如病例组样本少则权重高）。
# None：默认均等权重。
# 作用：
# 在病例-对照组不平衡时，balanced可提升少数类识别能力。
# 科研意义：验证是否需要对不平衡数据进行补偿。

# estimator
# 值：RandomForestClassifier(random_state=42)
# 作用：指定待优化的基模型，固定随机种子保证实验可复现。

# param_grid
# 值：上文中定义的参数网格。
# 作用：指定需要搜索的超参数组合空间。

# cv=5
# 将训练集分为5个子集。
# 依次用4个子集训练，1个子集验证，共5次。
# 综合5次结果评估参数性能。

# scoring='roc_auc'
# 含义：使用AUC（ROC曲线下面积）作为评估指标。
# 选择原因：
# AUC对类别不平衡不敏感。
# 衡量模型整体排序能力（将病例排在对照前的概率）。

# 获取最佳模型
best_rf = grid_search.best_estimator_
print(f"最佳参数组合: {grid_search.best_params_}")

# 5.3 使用最佳模型预测
y_pred = best_rf.predict(X_test_scaled)  # 直接预测类别
y_proba = best_rf.predict_proba(X_test_scaled)[:, 1]  # 预测正类概率
# predict()方法
# 输入特征数据，输出预测的类别标签（二分类中为0或1）
# 例：一维数组[0, 1, 0, 1, 1, ...]（0=对照组，1=病例组）
# predict_proba()方法
# 输入特征数据，输出每个样本属于各个类别的概率
# 例：二维数组[[0.8, 0.2], [0.1, 0.9], ...]（第一列=类别0的概率，第二列=类别1的概率）
# [:, 1]：提取所有样本属于类别1（病例组）的概率，生成一维数组y_proba

# X_test_scaled：已标准化的测试集特征数据。
# best_rf：通过网格搜索（GridSearchCV）得到的最优随机森林模型。

# 5.4 模型评估
# 基础指标
print("\n测试集准确率: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("测试集AUC: {:.3f}".format(roc_auc_score(y_test, y_proba)))
# eg.输出结果：
# 测试集AUC: 0.872
# AUC=0.872表明模型有87.2%的概率能够正确区分病例组和对照组样本(保留三位小数)

# 模型评估指标
# 基于类别标签的指标：
    # 准确率：accuracy_score(y_test, y_pred)
    # 混淆矩阵：confusion_matrix(y_test, y_pred)
    # 分类报告：classification_report(y_test, y_pred)（含精确率、召回率、F1值）
# 基于概率的指标：
    # AUC-ROC：roc_auc_score(y_test, y_proba)
    # ROC曲线：roc_curve(y_test, y_proba)
# 详见幕布https://www.mubu.com/doc/3HpLB9uWX2P

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

# confusion_matrix(y_test, y_pred)
# 功能：计算混淆矩阵
# 一个N×N矩阵（N为类别数），在二分类中为2×2矩阵，结构如下：
# cm = [TN FN]
#      [FP TP]  # 连成矩阵的大方框

# y_test：测试集的真实标签
# y_pred：模型的预测标签

# sns.heatmap()
# 功能：用热力图可视化混淆矩阵。
# 关键参数：
# cm：混淆矩阵数据。
# annot=True：在热力格中显示数值。
# fmt='d'：数值显示格式为整数。
# cmap='Blues'：颜色映射方案（蓝-白渐变）。

# 分类报告
print("\n分类报告:\n", classification_report(y_test, y_pred))
# 作用：生成分类模型的性能评估报告，包含精确率（Precision）、召回率（Recall）、F1分数（F1-Score）等核心指标。
# 输出：结构化表格，按类别和综合指标展示模型表现。
# 详见幕布https://www.mubu.com/doc/3HpLB9uWX2P

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
# 计算ROC曲线的假阳性率(FPR)、真阳性率(TPR)和阈值
# 计算ROC曲线下面积(AUC)
# fpr (False Positive Rate)：假阳性率数组
# tpr (True Positive Rate)：真阳性率数组
# 详见幕布https://www.mubu.com/doc/3HpLB9uWX2P

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

# 深橙色曲线与深蓝色虚线对比，突出模型性能优于随机猜测。
# 加粗曲线（lw=2）增强图表在论文或报告中的可读性。
# 动态标签显示AUC值，量化模型性能

# label='ROC曲线 (AUC = %0.2f)' % roc_auc
# label：图例标签，动态插入AUC值（%0.2f保留两位小数）
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# # [0, 1]：x和y轴的起点与终点，生成从(0,0)到(1,1)的直线
# 绘制随机猜测基线
# 代表随机猜测模型的性能（AUC=0.5），用于对比实际模型的优劣。
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# 标准化坐标轴，确保不同模型的ROC曲线可比较。

# 5.7 特征重要性分析
importances = best_rf.feature_importances_  # 提取模型学习到的特征重要性得分（数值化评估特征贡献）
indices = np.argsort(importances)[::-1]  # 生成按重要性 降序排列 的索引，用于后续可视化或分析时的特征排序

# 语法：model.feature_importances_
# 对象：
# best_rf：通过网格搜索或交叉验证得到的最优随机森林模型。
# 返回值：
# importances：一维数组，每个元素对应输入特征的重要性得分（总和为1）。
# 计算原理：
# 随机森林通过统计每棵树中特征的 分裂收益（Gini增益或信息增益），取所有树的平均值。
# 重要性越高，表示该特征对模型预测的贡献越大。

# 分解步骤：
# 排序索引获取：np.argsort(importances)
# 功能：返回数组值 从小到大 排序的 索引位置。
# 示例：若importances = [0.1, 0.3, 0.6]，则np.argsort返回[0, 1, 2]对应原始索引。
# 反向切片：[::-1]
# 功能：将数组 反向排列，实现 从大到小 排序。
# 示例：[0, 1, 2] → [2, 1, 0]。
# 最终结果：
# indices：存储按特征重要性 从高到低 排列的索引。
# 示例：若特征重要性为[0.6, 0.3, 0.1]，则indices = [0, 1, 2]；若重要性为[0.3, 0.6, 0.1]，则indices = [1, 0, 2]。

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




# plt.figure(figsize=(10, 8))  # 加大高度适应更多特征
# plt.title("随机森林特征重要性排序", fontsize=14, pad=20)
# plt.xlabel("重要性得分", fontsize=12, labelpad=10)
# # 绘制水平条形图（注意 y 轴反转排序）
# plt.barh(range(X_train.shape[1]), importances[indices],
#          color="steelblue", height=0.8)  # 调整颜色和条宽
# # 设置Y轴标签（特征名）
# plt.yticks(range(X_train.shape[1]),
#            X_train.columns[indices],
#            fontsize=10)  # 缩小字体适应长文本
# plt.gca().invert_yaxis()  # 反转Y轴，使最重要的特征显示在顶部
# plt.grid(axis='x', linestyle='--', alpha=0.6)  # 添加辅助网格线
# plt.tight_layout()
# plt.show()

# 标题与轴标签：
# 使用 plt.title() plt.xlabel() plt.ylabel() 明确图表含义
# 参数 fontsize 控制字体大小，labelpad 调整标签与轴的距离
# 水平条形图优势：
# 避免长特征名重叠（无需旋转标签）
# 通过 invert_yaxis() 让重要性排序更直观（从上到下递减）
# 可视化增强：
# grid() 添加网格线提高可读性
# color 参数使用更专业的色系（如 steelblue）
# 排序逻辑：
# 确保 indices = np.argsort(importances)[::-1] 特征已按重要性降序排列