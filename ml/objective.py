# Custom objective function for bayesian optimization
import numpy as np
from model import bench

input_transformer = lambda x: x
output_transformer = lambda x: x

def set_input_transformer(input_transformer_arg):
    global input_transformer
    input_transformer = input_transformer_arg

def set_output_transformer(output_transformer_arg):
    global output_transformer
    output_transformer = output_transformer_arg


# for DYCORS initial design fit
exp_design = []
exp_design_eval_count = 0

def set_initial_design(y):
    global exp_design
    exp_design = y

    print(exp_design)

def clear_initial_design():
    global exp_design
    exp_design = []
    global exp_design_eval_count
    exp_design_eval_count = 0

def objective(x):
    if isinstance(x, np.ndarray):
        x = x.flatten()

    x = input_transformer(x)

    global exp_design
    global exp_design_eval_count

    if exp_design_eval_count < len(exp_design):
        exp_design_eval_count += 1
        print("Found value in initial design")
        print(exp_design[exp_design_eval_count - 1][0])
        return exp_design[exp_design_eval_count - 1][0]

    # benchmark
    # return_value = bench(x)
    try:
        return_value = bench(x)
    except Exception as e:
        print("Error in benchmark:", e)
        return 0.0

    return_value = output_transformer(return_value)
    return return_value