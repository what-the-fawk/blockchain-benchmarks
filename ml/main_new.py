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
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression, f_regression 
from sklearn.decomposition import PCA
import shap
from collections.abc import Mapping
import re
from SALib import ProblemSpec
from sklearn.decomposition import PCA

from ml.feature_importance import (
    shap_feature_importance,
    random_forest_feature_importance,
    lasso_feature_importance,
    permutation_feature_importance,
    gradient_boosting_feature_importance
)

from ml.apply import (
    read_extract_config,
    read_configs,
    apply,
    add_context,
    get_context
)

from ml.model import (
    define_model,
    run_model
)


# 
def main():

    # future: start network

    domain = read_configs([]) # ../networks/*

    opt = define_model(
        domain=domain,
        context_list=None
    )

    run_model(domain, opt, [])


    pass

if __name__ == "__main__":
    from ruamel.yaml import YAML
    main()