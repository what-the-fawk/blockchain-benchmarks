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

from optimizer import BayesianOptimizer

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

from objective import objective

def clear_artefacts():
    artefacts_folder = os.path.join(SRC_FOLDER)
    for file_name in os.listdir(artefacts_folder):
        file_path = os.path.join(artefacts_folder, file_name)
        if os.path.isfile(file_path) and file_name != "README.md":
            os.remove(file_path)


from copy import deepcopy
from model import bench
from domain import restore_domain

def main():

    AF = None
    DR = None

    if "-a" in sys.argv:
        flag_index = sys.argv.index("-a")
        if flag_index + 1 < len(sys.argv):
            flag_value = sys.argv[flag_index + 1]
            AF = flag_value
        else:
            raise ValueError("Flag -a provided but no value specified.")
    else:
        raise ValueError("Flag -a not provided.")

    if "-d" in sys.argv:
        flag_index = sys.argv.index("-d")
        if flag_index + 1 < len(sys.argv):
            flag_value = sys.argv[flag_index + 1]
            DR = flag_value
        else:
            raise ValueError("Flag -d provided but no value specified.")
    else:
        raise ValueError("Flag -m not provided.")

    clear_artefacts()

    set_model_type('BO') # legacy

    read_configs([
        "../networks/configtx/configtx.yaml",
        "../networks/compose/docker/peercfg-org1/core.yaml",
        "../networks/compose/docker/peercfg-org2/core.yaml",
        "../networks/compose/docker/peercfg-org3/core.yaml",
        "../networks/compose/docker/peercfg-org4/core.yaml",
        # "../networks/compose/docker/ordcfg/orderer.yaml",
    ])
    
    domain = deepcopy(controllable_params)
    assert len(domain) > 0, "Domain is empty"
    print("Domain:", len(domain))

    # extract experimental design
    X = pd.read_csv('../design/X_initial', header=None)
    X.columns = ['feature_' + str(i) for i in range(X.shape[1])]
    X = X.iloc[:, :-1]
    X = X.iloc[:len(X.columns), :]

    y = pd.read_csv('../design/Y_initial', header=None).iloc[:len(X.columns), :]
    y.columns = ['target']

    # calibrate y
    pointx = X.iloc[0, :]
    pointy = y.iloc[0, :]

    ys = []

    for _ in range(10):
        point = bench(pointx)
        ys.append(point)

    average_y = np.mean([val for val in ys if val != min(ys)])
    print(ys)
    print(average_y)

    # calibrate values
    ratio = average_y / pointy
    print("Ratio:", ratio)
    print("Noise:", np.std(ys))

    # multiply column target by ratio
    target = y['target'].tolist()
    calibrated_y = [item * ratio for item in target]
    y = pd.DataFrame(calibrated_y, columns=['target'])

    # create and run model
    optimizer = BayesianOptimizer(
        X=X,
        y=y,
        domain=domain,
        AF=AF,
        DR=DR,
        n_iter=MAX_EVALUATIONS,
        n_relevant_features=15
    )

    optimizer.run()


if __name__ == "__main__":
    # from ruamel.yaml import YAML
    main()