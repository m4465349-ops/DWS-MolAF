#分类  输入的文件要有列名
# 最后生成的文件是排好序的
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import shap
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # 忽略SHAP警告

# 1. 数据准备
data = pd.read_csv(r"C:\Users\DELL\Desktop\二分类任务\Muta_2513.csv")
X = data.iloc[:, 1:].values  # 转换为numpy数组避免索引问题
y = data.iloc[:, 0].values
feature_names = data.columns[1:]

# 2. 划分数据集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Hepa
# params = {
#     'objective': 'binary',
#     'metric': ['auc', 'binary_logloss'],
#     'boosting_type': 'gbdt',
#     'seed': 42,
#     'verbosity': -1,
#     'n_jobs': -1,
#
#     # 树结构控制 - 针对2889条数据优化
#     'num_leaves': 15,      # 保持15，适合中等数据量
#     'max_depth': 4,        # 保持4，防止过拟合
#     'min_data_in_leaf': 15,  # 从20降低到15，增加模型灵活性
#     'max_bin': 127,        # 从63提高到127，增加特征离散化精度
#
#     # 学习率与正则化
#     'learning_rate': 0.05,  # 保持0.05
#     'lambda_l1': 0.2,      # 从0.3降低到0.2，稍微减少正则化
#     'lambda_l2': 0.2,      # 从0.3降低到0.2
#     'min_split_gain': 0.02, # 保持0.02
#
#     # 随机化参数
#     'feature_fraction': 0.7,  # 从0.5提高到0.7，利用更多特征
#     'bagging_fraction': 0.8,  # 从0.6提高到0.8，利用更多数据
#     'bagging_freq': 5,        # 从3提高到5，更频繁的bagging
#     'extra_trees': True,      # 保持启用
# }
# Muta
params = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'boosting_type': 'gbdt',
    'seed': 42,
    'verbosity': -1,
    'n_jobs': -1,

    # 树结构控制 - 针对8000条数据优化
    'num_leaves': 31,       # 从15增加到31，利用更多数据
    'max_depth': 6,         # 从4增加到6，增加模型容量
    'min_data_in_leaf': 20, # 从15增加到20，防止过拟合
    'max_bin': 255,         # 从127增加到255，提高特征精度

    # 学习率与正则化
    'learning_rate': 0.05,  # 保持0.05（可考虑降到0.03如果过拟合）
    'lambda_l1': 0.1,       # 从0.2降低到0.1，减少正则化
    'lambda_l2': 0.1,       # 从0.2降低到0.1
    'min_split_gain': 0.01, # 从0.02降低到0.01

    # 随机化参数
    'feature_fraction': 0.8,  # 从0.7提高到0.8，利用更多特征
    'bagging_fraction': 0.8,  # 保持0.8
    'bagging_freq': 3,        # 从5降低到3，减少随机性
    'extra_trees': False,     # 从True改为False，增加稳定性
}

# Cardio
# params = {
#     'objective': 'binary',
#     'metric': ['auc', 'binary_logloss'],
#     'boosting_type': 'gbdt',
#     'seed': 42,
#     'verbosity': -1,
#     'n_jobs': -1,
#
#     # 树结构控制 - 针对1500条数据优化（大幅调低！）
#     'num_leaves': 12,        # 从31大幅降低到12，防止过拟合
#     'max_depth': 4,          # 从6降低到4，减少模型复杂度
#     'min_data_in_leaf': 25,  # 从20增加到25，增强防过拟合
#     'max_bin': 63,           # 从255大幅降低到63，减少计算量
#
#     # 学习率与正则化
#     'learning_rate': 0.03,   # 从0.05降低到0.03，稳定训练
#     'lambda_l1': 0.3,        # 从0.1增加到0.3，增强正则化
#     'lambda_l2': 0.3,        # 从0.1增加到0.3
#     'min_split_gain': 0.02,  # 从0.01增加到0.02
#
#     # 随机化参数
#     'feature_fraction': 0.6,  # 从0.8降低到0.6，减少特征使用
#     'bagging_fraction': 0.7,  # 从0.8降低到0.7，减少数据使用
#     'bagging_freq': 5,        # 从3增加到5，增加随机性防过拟合
#     'extra_trees': True,      # 从False改为True，增加随机性
# }
# Carcin
# params = {
#     'objective': 'binary',
#     'metric': ['auc', 'binary_logloss'],
#     'boosting_type': 'gbdt',
#     'seed': 42,
#     'verbosity': -1,
#     'n_jobs': -1,
#
#     # 树结构控制 - 针对1021条数据优化（非常保守！）
#     'num_leaves': 8,  # 从12降低到8，大幅防止过拟合
#     'max_depth': 3,  # 从4降低到3，极简树结构
#     'min_data_in_leaf': 30,  # 从25增加到30，强约束分裂
#     'max_bin': 63,  # 保持63，减少计算量
#
#     # 学习率与正则化
#     'learning_rate': 0.02,  # 从0.03降低到0.02，更稳定训练
#     'lambda_l1': 0.5,  # 从0.3增加到0.5，强正则化
#     'lambda_l2': 0.5,  # 从0.3增加到0.5
#     'min_split_gain': 0.03,  # 从0.02增加到0.03，提高分裂门槛
#
#     # 随机化参数
#     'feature_fraction': 0.5,  # 从0.6降低到0.5，使用更少特征
#     'bagging_fraction': 0.6,  # 从0.7降低到0.6，使用更少数据
#     'bagging_freq': 5,  # 保持5，增加随机性
#     'extra_trees': True,  # 保持True，增加随机性
# }


# 4. 交叉验证特征重要性
def safe_importance(model, importance_type):
    """处理feature_importance可能返回None的情况"""
    imp = model.feature_importance(importance_type=importance_type)
    return np.zeros(X.shape[1]) if imp is None else imp


def cross_val_feature_importance(X, y, params, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    split_importances = np.zeros(X.shape[1])
    gain_importances = np.zeros(X.shape[1])

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params, n_estimators=1000)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(0)
            ]
        )

        booster = model.booster_
        split_importances += safe_importance(booster, 'split')
        gain_importances += safe_importance(booster, 'gain')

    return split_importances / n_folds, gain_importances / n_folds


cv_split, cv_gain = cross_val_feature_importance(X_train, y_train, params)

# 5. 主模型训练
model = lgb.LGBMClassifier(**params, n_estimators=1000)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(50)
    ]
)


# 6. SHAP值计算
def get_shap_importance(model, X_val, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # 取正类的SHAP值

    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(1, -1)

    return np.abs(shap_values).mean(axis=0)


shap_importance = get_shap_importance(model, X_val, feature_names)


# 7. 特征重要性整合
def normalize(arr):
    """稳健标准化"""
    arr = np.array(arr)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-10)


# 标准化重要性
split_norm = normalize(cv_split)
gain_norm = normalize(cv_gain)
shap_norm = normalize(shap_importance)

# 动态权重计算
corr_matrix = np.corrcoef([split_norm, gain_norm, shap_norm])
shap_weight = 0.3 if corr_matrix[2, 0] < 0.5 or corr_matrix[2, 1] < 0.5 else 0.2
combined = 0.5 * gain_norm + 0.3 * split_norm + shap_weight * shap_norm

# 8. 选择Top X特征并按重要性排序
top_2513_idx = np.argsort(combined)[-425:][::-1]
selected_features = feature_names[top_2513_idx]

# 创建完整重要性结果表
result_df = pd.DataFrame({
    'Feature': feature_names,
    'Split_Importance': split_norm,
    'Gain_Importance': gain_norm,
    'SHAP_Importance': shap_norm,
    'Combined_Score': combined
}).sort_values('Combined_Score', ascending=False)

# 保存结果（已注释掉）
# importance_file = r'C:\Users\DELL\Desktop\Hepa-Importance100.csv'
selected_file = r'C:\Users\DELL\Desktop\Muta_LightGBM5_ECFP4.csv'

# 保存带重要性评分的结果（全部特征）
# result_df.to_csv(importance_file, index=False)

# 保存筛选后的数据集（Top XXX）- 关键修改：按重要性排序
selected_data = pd.concat([
    data.iloc[:, 0],  # 保留标签列
    pd.DataFrame(X[:, top_2513_idx], columns=selected_features)
], axis=1)

# 按特征重要性重新排序列（新增的关键修改）
sorted_columns = [selected_data.columns[0]] + list(selected_features)
selected_data = selected_data[sorted_columns]
# selected_data.to_csv(selected_file, index=False)


# 9. 特征分类统计（新增功能）
def count_feature_types(selected_indices, feature_names):
    """统计选中的特征在各类型中的分布"""
    # 定义特征类型范围
    maccs_indices = list(range(0, 167))  # 索引0到166: MACCS
    erg_indices = list(range(167, 608))  # 索引167-607: ErG
    pubchem_indices = list(range(608, 1489))  # 索引608-1488: PubChem
    ecfp4_indices = list(range(1489, 2513))  # 索引1489-2512: ECFP4

    maccs_count = 0
    erg_count = 0
    pubchem_count = 0
    ecfp4_count = 0

    for idx in selected_indices:
        if idx in maccs_indices:
            maccs_count += 1
        elif idx in erg_indices:
            erg_count += 1
        elif idx in pubchem_indices:
            pubchem_count += 1
        elif idx in ecfp4_indices:
            ecfp4_count += 1

    return {
        'MACCS': maccs_count,
        'ErG': erg_count,
        'PubChem': pubchem_count,
        'ECFP4': ecfp4_count
    }


# 统计Top 100特征的类型分布
feature_type_counts = count_feature_types(top_2513_idx, feature_names)

# 10. 增强版结果输出
print("=" * 80)
print(f"{'模型训练结果':^80}")
print("=" * 80)
print(f"最佳迭代轮数: {model.best_iteration_}")
print(f"验证集AUC: {model.best_score_['valid_0']['auc']:.4f}")
print(f"验证集Logloss: {model.best_score_['valid_0']['binary_logloss']:.4f}")

print("\n" + "=" * 80)
print(f"{'特征重要性分析':^80}")
print("=" * 80)
print("重要性指标相关性矩阵:")
print(f"  Split vs Gain: {corr_matrix[0, 1]:.3f}")
print(f"  Split vs SHAP: {corr_matrix[0, 2]:.3f}")
print(f"  Gain vs SHAP: {corr_matrix[1, 2]:.3f}")

print("\nTop 10特征详情:")
for i, row in result_df.head(10).iterrows():
    print(f"  {row['Feature']:50} | "
          f"综合得分: {row['Combined_Score']:.3f} | "
          f"Gain: {row['Gain_Importance']:.3f} | "
          f"SHAP: {row['SHAP_Importance']:.3f}")

print("\n" + "=" * 80)
print(f"{'Top 100特征类型统计':^80}")
print("=" * 80)
print(f"MACCS特征 (0-166):    {feature_type_counts['MACCS']:3d} 个")
print(f"ErG特征 (167-607):    {feature_type_counts['ErG']:3d} 个")
print(f"PubChem特征 (608-1488): {feature_type_counts['PubChem']:3d} 个")
print(f"ECFP4特征 (1489-2512): {feature_type_counts['ECFP4']:3d} 个")
print(f"总计:                {sum(feature_type_counts.values()):3d} 个")

print("\n" + "=" * 80)
print(f"{'文件保存路径':^80}")
print("=" * 80)
# print(f"完整特征重要性文件: {importance_file}")
# print(f"Top特征数据集: {selected_file}")
print("=" * 80)
#


# #回归 排好序的
# #回归任务
# import lightgbm as lgb
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, KFold
# import shap
# import warnings
# from sklearn.metrics import mean_squared_error, mean_absolute_error
#
# warnings.filterwarnings("ignore", category=UserWarning)
#
# # 1. 数据准备
# # data = pd.read_csv(r"C:\Users\DELL\Desktop\回归任务\intraperitoneal_2513.csv")  # 第一列LD50，其余为特征
# data = pd.read_csv(r"C:\Users\DELL\Desktop\回归任务\intravenous_2513.csv")
# # data = pd.read_csv(r"C:\Users\DELL\Desktop\回归任务\oral_2513.csv")
#
# X = data.iloc[:, 1:].values
# y = data.iloc[:, 0].values
# feature_names = data.columns[1:]
#
# # 2. 划分数据集（回归任务无需stratify）
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=42)
#
# # 3. 回归专用参数
# params = {
#     # 核心参数
#     'objective': 'regression',
#     'metric': ['l2', 'l1'],          # 监控MSE和MAE
#     'boosting_type': 'gbdt',
#     'seed': 42,
#     'verbosity': -1,
#     'n_jobs': -1,
#
#     # 树结构控制（适配36k样本）
#     'num_leaves': 31,
#     'max_depth': 6,
#     'min_data_in_leaf': 100,
#     'max_bin': 255,                  # 增大以适应连续值
#
#     # 学习率与正则化
#     'learning_rate': 0.03,
#     'lambda_l1': 0.5,
#     'lambda_l2': 0.5,
#     'min_split_gain': 0.01,
#
#     # 随机化
#     'feature_fraction': 0.7,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'extra_trees': False
# }
#
# # 4. 工具函数
# def safe_importance(model, importance_type):
#     """处理feature_importance可能返回None的情况"""
#     imp = model.feature_importance(importance_type=importance_type)
#     return np.zeros(X.shape[1]) if imp is None else imp
#
# def normalize(arr):
#     """稳健标准化"""
#     arr = np.array(arr)
#     return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-10)
#
# # 5. 交叉验证特征重要性
# def cross_val_feature_importance(X, y, params, n_folds=5):
#     kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
#     split_importances = np.zeros(X.shape[1])
#     gain_importances = np.zeros(X.shape[1])
#
#     for train_idx, val_idx in kf.split(X):
#         X_train_fold, X_val_fold = X[train_idx], X[val_idx]
#         y_train_fold, y_val_fold = y[train_idx], y[val_idx]
#
#         model = lgb.LGBMRegressor(**params, n_estimators=1000)
#         model.fit(
#             X_train_fold, y_train_fold,
#             eval_set=[(X_val_fold, y_val_fold)],
#             callbacks=[
#                 lgb.early_stopping(stopping_rounds=100, verbose=False),
#                 lgb.log_evaluation(0)
#             ]
#         )
#         booster = model.booster_
#         split_importances += safe_importance(booster, 'split')
#         gain_importances += safe_importance(booster, 'gain')
#
#     return split_importances / n_folds, gain_importances / n_folds
#
# cv_split, cv_gain = cross_val_feature_importance(X_train, y_train, params)
#
# # 6. 主模型训练
# model = lgb.LGBMRegressor(**params, n_estimators=1000)
# model.fit(
#     X_train, y_train,
#     eval_set=[(X_val, y_val)],
#     callbacks=[
#         lgb.early_stopping(stopping_rounds=100, verbose=False),
#         lgb.log_evaluation(50)
#     ]
# )
#
# # 7. SHAP值计算
# def get_shap_importance(model, X_val, feature_names):
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X_val)
#     return np.abs(shap_values).mean(axis=0)
#
# shap_importance = get_shap_importance(model, X_val, feature_names)
#
# # 8. 特征重要性整合（逻辑不变）
# split_norm = normalize(cv_split)
# gain_norm = normalize(cv_gain)
# shap_norm = normalize(shap_importance)
#
# # 动态权重计算（与分类版本相同）
# corr_matrix = np.corrcoef([split_norm, gain_norm, shap_norm])
# shap_weight = 0.3 if corr_matrix[2, 0] < 0.5 or corr_matrix[2, 1] < 0.5 else 0.2
# combined = 0.5 * gain_norm + 0.3 * split_norm + shap_weight * shap_norm
#
# # 9. 选择Top特征（数量可调）
# top_n = 2513  # 根据需要修改
# top_idx = np.argsort(combined)[-top_n:][::-1]
# selected_features = feature_names[top_idx]
#
# # 10. 保存结果 - 修改部分：按重要性排序
# result_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Split_Importance': split_norm,
#     'Gain_Importance': gain_norm,
#     'SHAP_Importance': shap_norm,
#     'Combined_Score': combined
# }).sort_values('Combined_Score', ascending=False)
#
# # 新增：特征分类统计函数
# def count_feature_types(selected_indices, feature_names):
#     """统计选中的特征在各类型中的分布"""
#     # 定义特征类型范围
#     maccs_indices = list(range(0, 167))  # 索引0到166: MACCS
#     erg_indices = list(range(167, 608))  # 索引167-607: ErG
#     pubchem_indices = list(range(608, 1489))  # 索引608-1488: PubChem
#     ecfp4_indices = list(range(1489, 2513))  # 索引1489-2512: ECFP4
#
#     maccs_count = 0
#     erg_count = 0
#     pubchem_count = 0
#     ecfp4_count = 0
#
#     for idx in selected_indices:
#         if idx in maccs_indices:
#             maccs_count += 1
#         elif idx in erg_indices:
#             erg_count += 1
#         elif idx in pubchem_indices:
#             pubchem_count += 1
#         elif idx in ecfp4_indices:
#             ecfp4_count += 1
#
#     return {
#         'MACCS': maccs_count,
#         'ErG': erg_count,
#         'PubChem': pubchem_count,
#         'ECFP4': ecfp4_count
#     }
#
# # 统计Top特征的类型分布
# feature_type_counts = count_feature_types(top_idx, feature_names)
#
# # 输出文件
# # importance_file = r'C:\Users\DELL\Desktop\intraperitoneal_Importance.csv'
# # selected_file = r'C:\Users\DELL\Desktop\intraperitoneal-2200.csv'
# # importance_file = r'C:\Users\DELL\Desktop\intravenous_Importance.csv'
# selected_file = r'C:\Users\DELL\Desktop\intravenous-2513.csv'
# # importance_file = r'C:\Users\DELL\Desktop\oral_Importance.csv'
# # selected_file = r'C:\Users\DELL\Desktop\oral-2200.csv'
#
# # result_df.to_csv(importance_file, index=False)
#
# # 修改：保存筛选后的数据集并按重要性排序
# selected_data = pd.concat([
#     data.iloc[:, 0],  # 保留目标变量列
#     pd.DataFrame(X[:, top_idx], columns=selected_features)
# ], axis=1)
#
# # 按特征重要性重新排序列
# sorted_columns = [selected_data.columns[0]] + list(selected_features)
# selected_data = selected_data[sorted_columns]
# selected_data.to_csv(selected_file, index=False)
#
# # 11. 结果输出 - 增强版
# print("=" * 80)
# print(f"{'回归模型训练结果':^80}")
# print("=" * 80)
# print(f"最佳迭代轮数: {model.best_iteration_}")
# print(f"验证集MSE: {model.best_score_['valid_0']['l2']:.4f}")
# print(f"验证集MAE: {model.best_score_['valid_0']['l1']:.4f}")
#
# print("\n" + "=" * 80)
# print(f"{'特征重要性分析':^80}")
# print("=" * 80)
# print(f"重要性指标相关性矩阵:")
# print(f"  Split vs Gain: {corr_matrix[0, 1]:.3f}")
# print(f"  Split vs SHAP: {corr_matrix[0, 2]:.3f}")
# print(f"  Gain vs SHAP: {corr_matrix[1, 2]:.3f}")
#
# print("\nTop 10特征详情:")
# for i, row in result_df.head(10).iterrows():
#     print(f"  {row['Feature']:50} | 综合得分: {row['Combined_Score']:.3f}")
#
# print("\n" + "=" * 80)
# print(f"{f'Top {top_n}特征类型统计':^80}")
# print("=" * 80)
# print(f"MACCS特征 (0-166):    {feature_type_counts['MACCS']:3d} 个")
# print(f"ErG特征 (167-607):    {feature_type_counts['ErG']:3d} 个")
# print(f"PubChem特征 (608-1488): {feature_type_counts['PubChem']:3d} 个")
# print(f"ECFP4特征 (1489-2512): {feature_type_counts['ECFP4']:3d} 个")
# print(f"总计:                {sum(feature_type_counts.values()):3d} 个")
#
# print("\n" + "=" * 80)
# # print(f"已保存文件:\n- 完整特征重要性: {importance_file}\n- 筛选后数据集: {selected_file}")
# print("=" * 80)