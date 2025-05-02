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