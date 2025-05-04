import GPyOpt
import numpy as np
from domain import apply, transform_domain
from globals import SRC_FOLDER, MAX_EVALUATIONS, get_model_type
from feature_importance import shap_feature_importance, random_forest_feature_importance, lasso_feature_importance, permutation_feature_importance, gradient_boosting_feature_importance
from network_manager import start_network, start_benchmark, stop_network, observe_data
from pySOT.optimization_problems import OptimizationProblem
from poap.controller import SerialController

from pySOT.strategy import DYCORSStrategy
from pySOT.experimental_design import LatinHypercube
from pySOT.surrogate import RBFInterpolant  
import pandas as pd

MODEL_SPEC = ""

def bench(X):
    #TODO: check how data with context is passed here
    model_type = get_model_type()
    if model_type == 'BO':
        X = X[0]
    elif model_type == 'DYCORS':
        X = X
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # save point to file
    with open(SRC_FOLDER + get_model_type() + "_" + MODEL_SPEC + "_points", "a") as f:
        f.write("".join(map(lambda x: str(x) + ",", X)) + "\n")

    apply(X)
    start_network()
    start_benchmark()
    stop_network()
    return observe_data()

def define_model(domain, **kwargs):

    opt = None
    dim = len(domain)
    initial_points = dim + 1
    model_type = get_model_type()

    if model_type == 'BO':

        global MODEL_SPEC
        MODEL_SPEC = kwargs.get('model_spec', "")

        opt = GPyOpt.methods.BayesianOptimization(
            f=bench,
            domain=transform_domain(domain),
            acquisition_type='EI',
            initial_design_type='latin',
            initial_design_numdata=initial_points,
            maximize=True,
            num_cores=6,
            X=kwargs.get('X', None),
            Y=kwargs.get('y', None),
        )

    elif model_type == 'DYCORS':

        print('creating dycors')

        class CustomProblem(OptimizationProblem): 
            def __init__(self, dim):
                self.dim = dim
                self.lb = np.array([item['bounds'][0] for item in domain])
                self.ub = np.array([item['bounds'][-1] for item in domain])
                self.cont_var = np.array([i for i, item in enumerate(domain) if item['type'] == 'continuous'])
                self.int_var = np.array([i for i, item in enumerate(domain) if item['type'] == 'discrete'])
                self.info = "Hyperledger Fabric Tuning"

            def eval(self, x):
                return bench(x)

        
        problem = CustomProblem(dim=dim)
        print('Define custom problem')

        strategy = DYCORSStrategy(
            max_evals=MAX_EVALUATIONS,
            opt_prob=problem,
            exp_design=LatinHypercube(dim=dim, num_pts=initial_points),
            surrogate=RBFInterpolant(dim=dim, lb=problem.lb, ub=problem.ub),
            num_cand=2,
        )
        print('Define strategy')
        opt = SerialController(objective=problem.eval)
        opt.strategy = strategy
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return opt

#TODO: my acquisition function

#TODO: check 'mutual_info', 'f_regression'
#TODO: scale features in feature importance evaluation (StandardScaler)

# feature importance methods = ['shap', 'random_forest', 'lasso', 'permutation', 'gradient_boosting']

#TODO: future:
# new params for run_model - X, y - to train model after feature importance

# def select_features(X, y) -> List[int]: --> most relevant features according to algorithms

def run_model(opt: GPyOpt.methods.BayesianOptimization | SerialController, context = None, **kwargs):
    # raise RuntimeError("better yaml package unused")
    X, y = None, None
    model_type = get_model_type()

    if model_type == 'BO':
        num_epochs = kwargs.get('num_epochs', MAX_EVALUATIONS)

        opt.run_optimization(max_iter=num_epochs, context=context, report_file=SRC_FOLDER + "BO_report_fullspace", evaluations_file=SRC_FOLDER + "BO_evals_fullspace")
        X, y = opt.get_evaluations()

    elif model_type == 'DYCORS':
        opt.run()
        X, y = opt.strategy.X, opt.strategy.fX
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    columns = [f"{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=columns)
    # print(X, "\n", y)

    return X_df, y
    
    #TODO: output X, y to define models for feature importance
    
    # print(importance_df)
    # raise RuntimeError("feature importance not implemented yet")

