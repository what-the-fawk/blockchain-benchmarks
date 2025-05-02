from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
np.bool = np.bool_
import shap

# 1. SHAP Feature importance
def shap_feature_importance(X, y):
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_sum = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({'Feature': X.columns, 'SHAP Importance': shap_sum})
    shap_importance = shap_importance.sort_values(by='SHAP Importance', ascending=False)
    return shap_importance

# 2. Random Forest Feature Importance
def random_forest_feature_importance(X, y):
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    rf_importance = pd.DataFrame({'Feature': X.columns, 'RF Importance': model.feature_importances_})
    rf_importance = rf_importance.sort_values(by='RF Importance', ascending=False)
    return rf_importance

# 3. L1 Regularization (Lasso) Feature Importance
def lasso_feature_importance(X, y):
    model = Lasso(alpha=0.00001, max_iter=10000)
    model.fit(X, y)
    lasso_importance = pd.DataFrame({'Feature': X.columns, 'Lasso Coefficient': np.abs(model.coef_)})
    lasso_importance = lasso_importance.sort_values(by='Lasso Coefficient', ascending=False)
    return lasso_importance

# 4. Permutation Feature Importance
def permutation_feature_importance(X, y):
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    perm_importance_df = pd.DataFrame({'Feature': X.columns, 'Permutation Importance': perm_importance.importances_mean})
    perm_importance_df = perm_importance_df.sort_values(by='Permutation Importance', ascending=False)
    return perm_importance_df

# 5. Gradient Boosting Feature Importance
def gradient_boosting_feature_importance(X, y):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    gb_importance = pd.DataFrame({'Feature': X.columns, 'GB Importance': model.feature_importances_})
    gb_importance = gb_importance.sort_values(by='GB Importance', ascending=False)
    return gb_importance

# TODO: PCA, sensitivity analysis, etc.
# def sensitivity_analysis_feature_importance(X, y):
#     model = RandomForestRegressor()
#     model.fit(X, y)

#     problem = {
#         'num_vars': X.shape[1],
#         'names': X.columns.tolist(),
#         'bounds': [[X[col].min(), X[col].max()] for col in X.columns]
#     }

#     sp = ProblemSpec(problem)
#     sp.sample_saltelli(1000)

#     sp.set_results(model.predict(sp.samples))
#     sp.analyze_sobol()

#     sobol_df = pd.DataFrame({
#         'Feature': problem['names'],
#         'S1': sp.analysis['S1'],
#         'ST': sp.analysis['ST']
#     }).sort_values(by='ST', ascending=False)

#     return sobol_df

# def pca_feature_importance(X, y):
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X, y)

#     loadings = pd.DataFrame(
#             np.abs(pca.components_),
#             columns=X.columns
#         )

#     explained_variance = pca.explained_variance_ratio_
#     weighted_loadings = loadings.multiply(explained_variance, axis=0)
#     importance_scores = weighted_loadings.sum(axis=0).sort_values(ascending=False)

#     results = pd.DataFrame({
#         'Feature': importance_scores.index,
#         'Importance': importance_scores.values
#     }).reset_index(drop=True)

#     return results