import GPyOpt
import numpy as np
from apply import add_context, get_context
from feature_importance import shap_feature_importance
from network_manager import start_network, start_benchmark, stop_network, observe_data

def fit(X):
    # start benchmark
    start_network()
    start_benchmark()
    stop_network()

    return observe_data()
    pass

def define_model(domain, context_list):

    opt = GPyOpt.methods.BayesianOptimization(
        f=fit,
        domain=domain,
        fixed_features=context_list,
        acquisition_type='EI',
        maximize=True,
        num_cores=4
    )

    return opt

NUM_EPOCHS = 20

def run_model(domain, opt: GPyOpt.methods.BayesianOptimization, context_data):
    opt.run_optimization(max_iter=NUM_EPOCHS, context=context_data)

    X, y = opt.get_evaluations()

    shap_feature_importance(X, y)


