# POSSIBLE VALUES
# 'BO', 'DYCORS'
MODEL_TYPE = None

def set_model_type(model_type):
    print(f"Setting model type to: {model_type}")
    global MODEL_TYPE
    if model_type not in ['BO', 'DYCORS']:
        raise ValueError(f"Unknown model type: {model_type}")
    MODEL_TYPE = model_type

def get_model_type():
    return MODEL_TYPE


MAX_EVALUATIONS = 500
BO_INIT_EVALUATIONS = 300

SRC_FOLDER = '../artefacts/'