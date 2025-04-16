# 6. XGboost模型

# 6.1 数据格式转换
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)
# 使用DMatrix格式提升运算效率

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
# 最后一句   或使用交叉验证寻找最优权重：
# param_grid = {'scale_pos_weight': [1, 3, 5, 7, 9, 15]}

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
# 此时 cv_results 是一个 DataFrame
# 行索引	test-auc-mean	test-auc-std	train-auc-mean	...
# 0	    0.82	        0.03	        0.95	        ...
# 1  	0.85	        0.02	        0.96	        ...
# ...	...	            ...	            ...	            ...
# N	    0.92	        0.01	        0.99	        ...
# 每行对应一次迭代（boosting round）
# 列包含评估指标（如AUC的均值、标准差等）

# 获取最优迭代次数
best_rounds = cv_results.shape[0]
# shape 是 Pandas DataFrame和Numpy数组的共有属性：
    # 返回值：形如 (行数, 列数) 的元组
    # 示例：
    # print(cv_results.shape)  # 输出 (150, 6) 表示150行6列
# shape[0] 的实际意义
    # shape[0] 取元组第一个元素，即数据行数
    # 在交叉验证场景中：表示 实际执行的迭代次数
    # 若早停机制生效（如第180轮停止），则 shape[0]=180
    # 若未触发早停（完成全部500轮），则 shape[0]=500

# 6.4 模型训练
model_xgb = xgb.train(
    params,
    dtrain,
    num_boost_round=best_rounds,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    verbose_eval=50
)
# evals=[(dtrain, 'train'), (dtest, 'test')],
# 该参数用于 实时监控模型在不同数据集上的表现，主要实现：
# 训练过程可视化：输出每次迭代时指定数据集的评估指标
# 早停机制支持：配合early_stopping_rounds自动选择最优模型
# 过拟合检测：对比训练集和测试集表现的差异
# dtrain、dtest分别是经数据格式转换后的标准化训练集特征数据与测试集特征数据
# (dtrain, 'train')：训练数据集及其别名
# (dtest, 'test')：测试数据集及其别名

# 医疗数据分析场景示例
# 假设您的POPs与结肠癌数据：
# 训练集：2018-2021年患者数据（800例）
# 测试集：2022年患者数据（200例）
#
# 当设置：
# evals=[(dtrain, '2018-2021'), (dtest, '2022')]
# 训练时输出：
# [0]	2018-2021-auc:0.92	2022-auc:0.86
# [1]	2018-2021-auc:0.93	2022-auc:0.87
# ...
# [150]	2018-2021-auc:0.99	2022-auc:0.91
#
# 通过观察2022集的AUC变化，可以判断模型是否在新数据上保持良好表现。

# 6.5 模型预测
y_pred_prob = model_xgb.predict(dtest)  # 预测概率值
y_pred = (y_pred_prob > 0.5).astype(int)  # 概率转类别标签
# model_xgb.predict()	返回样本属于正类（癌症）的概率	量化患病风险水平
# 0.5	                分类阈值	                    默认临床诊断临界值
# astype(int)	        数据类型转换	                生成最终的诊断结果（0/1）
# 输出类型：y_pred_prob为连续概率值（0~1）
# 决策方式：硬性阈值划分（二分类）
# 应用场景：需要风险概率的后续分析（如风险分级）

# VS
#
# 5.3 使用最佳模型预测
# y_pred = best_rf.predict(X_test_scaled)  # 直接预测类别
# y_proba = best_rf.predict_proba(X_test_scaled)[:, 1]  # 预测正类概率
# predict()方法
# 输入特征数据，输出预测的类别标签（二分类中为0或1）
# 例：一维数组[0, 1, 0, 1, 1, ...]（0=对照组，1=病例组）
# predict_proba()方法
# 输入特征数据，输出每个样本属于各个类别的概率
# 例：二维数组[[0.8, 0.2], [0.1, 0.9], ...]（第一列=类别0的概率，第二列=类别1的概率）
# [:, 1]：提取所有样本属于类别1（病例组）的概率，生成一维数组y_proba

# X_test_scaled：已标准化的测试集特征数据。
# best_rf：通过网格搜索（GridSearchCV）得到的最优随机森林模型。
#
# 相同点
    # 维度	共同特性	                医疗应用价值
    # 目标	获取预测结果（类别+概率）	支持临床决策
    # 本质	基于训练好的模型进行推断	实现自动化诊断
    # 评估	均可计算AUC等指标	        模型效果验证
# 不同点
    # 维度	    XGBoost实现	        随机森林实现	        对医疗分析的影响
    # 预测方法	predict()返回概率	predict()返回类别	XGB需额外转换，RF更直接
    # 概率获取	单次预测完成	        需调用predict_proba	RF需要更多计算步骤
    # 输出结构	单一概率数组	        分离的类别和概率	    数据处理流程差异
    # 模型特性	Boosting算法	        Bagging算法	        风险模式识别方式不同

# 对于predict()方法
# XGBoost	predict()	连续值	[0,1]	样本属于正类（癌症）的概率
# 随机森林	predict()	离散值	{0,1}	直接预测的类别标签

# XGBoost实现
# y_pred_prob = model_xgb.predict(dtest)  # → 概率值（如0.83）
# y_pred = (y_pred_prob > 0.5).astype(int)  # → 阈值划分（如1）
#
# 随机森林实现
# y_pred = best_rf.predict(X_test_scaled)  # → 直接输出类别（0/1）
# y_proba = best_rf.predict_proba(X_test_scaled)[:,1]  # → 概率值（如0.79）
#
# 差异产生原因
# 维度	        XGBoost设计逻辑	            随机森林设计逻辑	            对医疗数据分析的影响
# 预测默认值	默认返回概率（梯度提升特性）	    默认返回类别（Bagging投票机制）	XGB需手动转换，RF需显式获取概率
# 方法命名	    predict()=概率预测	        predict_proba()=概率预测	    需要注意API差异
# 业务场景	    更适合需要概率评估的精细分析	    更适合快速分类决策             XGB适合风险评估，RF适合初筛

# 6.6 模型评估
print("\n=== 模型评估结果 ===")
# 基础指标
print("测试集准确率: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("测试集AUC: {:.3f}".format(roc_auc_score(y_test, y_pred_prob)))
# eg.输出结果：
# 测试集AUC: 0.872
# AUC=0.872表明模型有87.2%的概率能够正确区分病例组和对照组样本(保留三位小数)

# VS

# 5.4 模型评估
# 基础指标
# print("\n测试集准确率: {:.3f}".format(accuracy_score(y_test, y_pred)))
# print("测试集AUC: {:.3f}".format(roc_auc_score(y_test, y_proba)))

# 一样的，准确率用的都是测试集标签与y_pred(类别标签)
# AUC用的都是测试集标签与y_pred_prob、y_proba(概率)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()
# 见5. 随机森林模型

# 分类报告
print("\n分类报告:\n", classification_report(y_test, y_pred))
# 见5. 随机森林模型

# 6.7 ROC曲线绘制
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
# 见5. 随机森林模型

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
# 见5. 随机森林模型

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
# plt.grid(True)：显示网格线
# plt.grid(False)：隐藏网格线

# 6.9 保存模型
model_xgb.save_model('xgb_model.json')  # 保存为JSON格式