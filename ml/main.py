import os
import subprocess
import yaml
import json
from bs4 import BeautifulSoup
import numpy as np
from skopt import Optimizer
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from line_profiler import LineProfiler
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression, f_regression 
from sklearn.decomposition import PCA # TODO
import shap
from collections.abc import Mapping
import re
from SALib import ProblemSpec
from sklearn.decomposition import PCA
import sys

from feature_importance import (
    shap_feature_importance,
    random_forest_feature_importance,
    lasso_feature_importance,
    permutation_feature_importance,
    gradient_boosting_feature_importance
)

from globals import (
    SRC_FOLDER,
    MAX_EVALUATIONS,
    BO_INIT_EVALUATIONS,
    set_model_type,
    get_model_type
)

from domain import (
    read_configs,
    apply,
    controllable_params,
    context,
    transform_domain,
    fix_domain
)

from model import (
    define_model,
    run_model
)

def clear_artefacts():
    artefacts_folder = os.path.join(SRC_FOLDER)
    for file_name in os.listdir(artefacts_folder):
        file_path = os.path.join(artefacts_folder, file_name)
        if os.path.isfile(file_path) and file_name != "README.md":
            os.remove(file_path)


# 
def main():

    if "-m" in sys.argv:
        m_flag_index = sys.argv.index("-m")
        if m_flag_index + 1 < len(sys.argv):
            m_flag_value = sys.argv[m_flag_index + 1]
            set_model_type(m_flag_value)
        else:
            raise ValueError("Flag -m provided but no value specified.")
    else:
        m_flag_value = None
        print("Flag -m not provided.")

    clear_artefacts()

    read_configs([
        "../networks/configtx/configtx.yaml",
        "../networks/compose/docker/peercfg-org1/core.yaml",
        "../networks/compose/docker/peercfg-org2/core.yaml",
        "../networks/compose/docker/peercfg-org3/core.yaml",
        "../networks/compose/docker/peercfg-org4/core.yaml",
        # "../networks/compose/docker/ordcfg/orderer.yaml",
    ])
    
    opt = define_model(domain=controllable_params)

    model_type = get_model_type()
    num_epochs = MAX_EVALUATIONS

    if model_type == 'BO':
        num_epochs = BO_INIT_EVALUATIONS

    X, y = run_model(opt, num_epochs=num_epochs)

    with open(SRC_FOLDER + model_type + "_X.csv", "w") as f:
        X.to_csv(f, index=False)
    with open(SRC_FOLDER + model_type + "_y.csv", "w") as f:
        pd.DataFrame(y).to_csv(f, index=False)

    max_y_index = np.argmax(y)
    best_X = X.iloc[max_y_index]

    best_point_file = SRC_FOLDER + model_type + "_best_point.json"
    with open(best_point_file, "w") as f:
        json.dump(best_X.to_dict(), f, indent=4)
    print(f"Best point saved to {best_point_file}")
    

    # Feature importance
    if model_type == 'BO':
        relevant_features_num = 20

        shap_top_features = shap_feature_importance(X, y).head(relevant_features_num)['Feature'].tolist()
        rf_top_features = random_forest_feature_importance(X, y).head(relevant_features_num)['Feature'].tolist()
        lasso_top_features = lasso_feature_importance(X, y).head(relevant_features_num)['Feature'].tolist()
        permutation_top_features = permutation_feature_importance(X, y).head(relevant_features_num)['Feature'].tolist()
        gb_top_features = gradient_boosting_feature_importance(X, y).head(relevant_features_num)['Feature'].tolist()

        # convert into GpyOpt X, y format
        X = X.to_numpy()
        y = np.array(y).reshape(-1, 1)

        shap_domain, shap_ctx = fix_domain(controllable_params, shap_top_features, best_X=best_X)
        opt = define_model(domain=shap_domain, X=X, y=y, model_spec="shap")
        X_shap, y_shap = run_model(opt, context=shap_ctx, num_epochs=MAX_EVALUATIONS - BO_INIT_EVALUATIONS)
        with open(SRC_FOLDER + model_type + "_shapX.csv", "w") as f:
            X_shap.to_csv(f, index=False)
        with open(SRC_FOLDER + model_type + "_shapy.csv", "w") as f:
            pd.DataFrame(y_shap).to_csv(f, index=False)

        rf_domain, rf_ctx = fix_domain(controllable_params, rf_top_features, best_X=best_X)
        opt = define_model(domain=rf_domain, X=X, y=y, model_spec="rf")
        X_rf, y_rf = run_model(opt, context=rf_ctx)

        with open(SRC_FOLDER + model_type + "_rfX.csv", "w") as f:
            X_rf.to_csv(f, index=False)
        with open(SRC_FOLDER + model_type  + "_rfy.csv", "w") as f:
            pd.DataFrame(y_rf).to_csv(f, index=False)

        lasso_domain, lasso_ctx = fix_domain(controllable_params, lasso_top_features, best_X=best_X)
        opt = define_model(domain=lasso_domain, X=X, y=y, model_spec="lasso")
        X_lasso, y_lasso = run_model(opt, context=lasso_ctx, num_epochs=MAX_EVALUATIONS - BO_INIT_EVALUATIONS)
        
        with open(SRC_FOLDER + model_type  + "_lassoX.csv", "w") as f:
            X_lasso.to_csv(f, index=False)
        with open(SRC_FOLDER + model_type  + "_lassoy.csv", "w") as f:
            pd.DataFrame(y_lasso).to_csv(f, index=False)

        permutation_domain, permutation_ctx = fix_domain(controllable_params, permutation_top_features, best_X=best_X)
        opt = define_model(domain=permutation_domain, X=X, y=y, model_spec="permutation")
        X_permutation, y_permutation = run_model(opt, context=permutation_ctx, num_epochs=MAX_EVALUATIONS - BO_INIT_EVALUATIONS)
        
        with open(SRC_FOLDER + model_type  + "_permutationX.csv", "w") as f:
            X_permutation.to_csv(f, index=False)
        with open(SRC_FOLDER + model_type  + "_permutationy.csv", "w") as f:
            pd.DataFrame(y_permutation).to_csv(f, index=False)

        gb_domain, gb_ctx = fix_domain(controllable_params, gb_top_features, best_X=best_X)
        opt = define_model(domain=gb_domain, X=X, y=y, model_spec="gb")
        X_gb, y_gb = run_model(opt, context=gb_ctx, num_epochs=MAX_EVALUATIONS - BO_INIT_EVALUATIONS)
        
        with open(SRC_FOLDER + model_type  + "_gbX.csv", "w") as f:
            X_gb.to_csv(f, index=False)
        with open(SRC_FOLDER + model_type  + "_gby.csv", "w") as f:
            pd.DataFrame(y_gb).to_csv(f, index=False)


if __name__ == "__main__":
    # from ruamel.yaml import YAML
    main()