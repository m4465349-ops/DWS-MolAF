import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import shap
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

data = pd.read_csv(r"Hepa_2513.csv")
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
feature_names = data.columns[1:]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

params = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'boosting_type': 'gbdt',
    'seed': 42,
    'verbosity': -1,
    'n_jobs': -1,
    'num_leaves': 15,
    'max_depth': 4,
    'min_data_in_leaf': 15,
    'max_bin': 127,
    'learning_rate': 0.05,
    'lambda_l1': 0.2,
    'lambda_l2': 0.2,
    'min_split_gain': 0.02,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'extra_trees': True,
}

def safe_importance(model, importance_type):
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

model = lgb.LGBMClassifier(**params, n_estimators=1000)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(50)
    ]
)

def get_shap_importance(model, X_val, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(1, -1)

    return np.abs(shap_values).mean(axis=0)

shap_importance = get_shap_importance(model, X_val, feature_names)

def normalize(arr):
    arr = np.array(arr)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-10)

split_norm = normalize(cv_split)
gain_norm = normalize(cv_gain)
shap_norm = normalize(shap_importance)

def calculate_dynamic_weights(split_norm, gain_norm, shap_norm):
    corr_matrix = np.corrcoef([split_norm, gain_norm, shap_norm])

    uniqueness_split = 1 - (corr_matrix[0, 1] + corr_matrix[0, 2]) / 2
    uniqueness_gain = 1 - (corr_matrix[1, 0] + corr_matrix[1, 2]) / 2
    uniqueness_shap = 1 - (corr_matrix[2, 0] + corr_matrix[2, 1]) / 2

    uniqueness_scores = np.array([uniqueness_gain, uniqueness_split, uniqueness_shap])
    weights = np.exp(uniqueness_scores) / np.sum(np.exp(uniqueness_scores))

    return weights[0], weights[1], weights[2], corr_matrix

gain_weight, split_weight, shap_weight, corr_matrix = calculate_dynamic_weights(split_norm, gain_norm, shap_norm)
combined = gain_weight * gain_norm + split_weight * split_norm + shap_weight * shap_norm

top_2513_idx = np.argsort(combined)[-20:][::-1]
selected_features = feature_names[top_2513_idx]

result_df = pd.DataFrame({
    'Feature': feature_names,
    'Split_Importance': split_norm,
    'Gain_Importance': gain_norm,
    'SHAP_Importance': shap_norm,
    'Combined_Score': combined
}).sort_values('Combined_Score', ascending=False)

selected_file = r'Hepa-LightGBM5-20.csv'

selected_data = pd.concat([
    data.iloc[:, 0],
    pd.DataFrame(X[:, top_2513_idx], columns=selected_features)
], axis=1)

sorted_columns = [selected_data.columns[0]] + list(selected_features)
selected_data = selected_data[sorted_columns]

selected_data.to_csv(selected_file, index=False)

def count_feature_types(selected_indices, feature_names):
    maccs_indices = list(range(0, 167))
    erg_indices = list(range(167, 608))
    pubchem_indices = list(range(608, 1489))
    ecfp4_indices = list(range(1489, 2513))

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

feature_type_counts = count_feature_types(top_2513_idx, feature_names)

print("=" * 80)
print(f"{'Model Training Results':^80}")
print("=" * 80)
print(f"Best iteration: {model.best_iteration_}")
print(f"Validation AUC: {model.best_score_['valid_0']['auc']:.4f}")
print(f"Validation Logloss: {model.best_score_['valid_0']['binary_logloss']:.4f}")

print("\n" + "=" * 80)
print(f"{'Feature Importance Analysis':^80}")
print("=" * 80)
print("Importance correlation matrix:")
print(f"  Split vs Gain: {corr_matrix[0, 1]:.3f}")
print(f"  Split vs SHAP: {corr_matrix[0, 2]:.3f}")
print(f"  Gain vs SHAP: {corr_matrix[1, 2]:.3f}")

print(f"\nDynamic weights:")
print(f"  Gain weight: {gain_weight:.3f}")
print(f"  Split weight: {split_weight:.3f}")
print(f"  SHAP weight: {shap_weight:.3f}")
print(f"  Total weight: {gain_weight + split_weight + shap_weight:.3f}")

print("\nTop 10 features:")
for i, row in result_df.head(10).iterrows():
    print(f"  {row['Feature']:50} | "
          f"Combined: {row['Combined_Score']:.3f} | "
          f"Gain: {row['Gain_Importance']:.3f} | "
          f"SHAP: {row['SHAP_Importance']:.3f}")

print("\n" + "=" * 80)
print(f"{'Top 20 Feature Type Statistics':^80}")
print("=" * 80)
print(f"MACCS features (0-166):    {feature_type_counts['MACCS']:3d}")
print(f"ErG features (167-607):    {feature_type_counts['ErG']:3d}")
print(f"PubChem features (608-1488): {feature_type_counts['PubChem']:3d}")
print(f"ECFP4 features (1489-2512): {feature_type_counts['ECFP4']:3d}")
print(f"Total:                {sum(feature_type_counts.values()):3d}")

print("\n" + "=" * 80)
print(f"{'File Save Path':^80}")
print("=" * 80)
print(f"Selected features dataset: {selected_file}")
print("=" * 80)