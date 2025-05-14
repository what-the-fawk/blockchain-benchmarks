
import GPyOpt
import numpy as np
import pandas as pd
from pySOT.surrogate import GPRegressor
from pySOT.strategy import DYCORSStrategy
from poap.controller import SerialController
from pySOT.optimization_problems import OptimizationProblem
from sklearn.gaussian_process import GaussianProcessRegressor
from pySOT.experimental_design.experimental_design import ExperimentalDesign

from domain import transform_domain, cut_domain
from objective import objective, set_initial_design, set_input_transformer, set_output_transformer
from feature_importance import pca_feature_importance, sa_feature_importance, shap_feature_importance

def init_DYCORS(domain, n_trials, X, y, dim, is_rembo = False):

    if is_rembo:
        # transform domain to be lower-dimsional. All variables are continuous and with no bounds
        domain = [{'type': 'continuous', 'bounds': [-1e4, 1e4]} for _ in range(dim)]
        # X and y are already transformed

        # check if row-vectors from X are in bounds, if not - clip
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i, j] < domain[j]['bounds'][0]:
                    X[i, j] = domain[j]['bounds'][0]
                elif X[i, j] > domain[j]['bounds'][1]:
                    X[i, j] = domain[j]['bounds'][1]

    class CustomProblem(OptimizationProblem):
        def __init__(self, dim, lb, ub):
            self.dim = dim
            self.lb = lb
            self.ub = ub
            self.cont_var = np.array([i for i, item in enumerate(domain) if item['type'] == 'continuous'])
            self.int_var = np.array([i for i, item in enumerate(domain) if item['type'] == 'discrete'])
            self.info = "Hyperledger Fabric Tuning"

        def eval(self, x):
            return objective(x)
            
    class PrecomputedDesign(ExperimentalDesign):
        def __init__(self, dim, X):
            self.dim = dim
            self.num_pts = X.shape[0]
            self.X = X

        def generate_points(self, lb, ub, int_var=None):
            return self.X
            

    lb = np.array([item['bounds'][0] for item in domain])
    ub = np.array([item['bounds'][1] for item in domain])

    regressor = GaussianProcessRegressor(random_state=13)
    regressor.fit(X, y)

    regressor_dycors = GPRegressor(dim=dim, lb=lb, ub=ub, gp=regressor)
    problem = CustomProblem(dim=dim, lb=lb, ub=ub)
    design = PrecomputedDesign(dim=dim, X=X)

    # model evaluation

    strat = DYCORSStrategy(max_evals=design.num_pts + n_trials,
                       surrogate=regressor_dycors,
                       opt_prob=problem,
                       exp_design=design,
                       asynchronous=False,
                       batch_size=1,
                       num_cand=1,
                       )
    opt = SerialController(objective=problem.eval)
    opt.strategy = strat

    return opt

def init_BO(domain, X, y, acquisition_type, dim, is_rembo=False): # EI or LCB or MPI

    if is_rembo:
        # transform domain to be lower-dimsional. All variables are continuous and with no bounds
        domain = [{'type': 'continuous', 'bounds': [-1e4, 1e4]} for _ in range(dim)]
        # X and y are already transformed

        # check if row-vectors from X are in bounds, if not - clip
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i, j] < domain[j]['bounds'][0]:
                    X[i, j] = domain[j]['bounds'][0]
                elif X[i, j] > domain[j]['bounds'][1]:
                    X[i, j] = domain[j]['bounds'][1]

    opt = GPyOpt.methods.BayesianOptimization(
            f=objective,
            domain=transform_domain(domain),
            acquisition_type=acquisition_type,
            initial_design_type='random', # placeholder
            initial_design_numdata=1,
            maximize=True,
            normalize_Y=True,
            num_cores=1,
            X=X,
            Y=y
    )
    return opt

class BayesianOptimizer:
    def __init__(self,
                X: pd.DataFrame,
                y: pd.DataFrame,
                domain: list,
                AF: str,
                DR: str,
                n_iter: int,
                n_relevant_features: int = 15,
                ):
        
        self.__check_params(AF, DR)
        
        self.X = X
        self.y = y
        self.AF = AF
        self.DR = DR
        self.domain = domain
        self.n_relevant_features = n_relevant_features
        self.model = None
        self.n_iter = n_iter

        self.__init_model()

    def __select_important_features(self):
        if self.DR == 'shap':
            return shap_feature_importance(self.X, self.y).head(self.n_relevant_features).index.to_list()
        elif self.DR == 'pca':
            return pca_feature_importance(self.X, self.y).head(self.n_relevant_features).index.to_list()
        elif self.DR == 'sa':
            return sa_feature_importance(self.X, self.y, [item['bounds'] for item in self.domain]).head(self.n_relevant_features).index.to_list()
        else:
            raise ValueError(f"Variable selection for {self.DR} is not supported")
        
    def __check_params(self, AF: str, DR: str):    
        if AF not in ['EI', 'UCB', 'MPI', 'DYCORS']:
            raise ValueError(f"Acquisition function {AF} not supported")
        if DR not in ['shap', 'pca', 'sa', 'rembo']:
            raise ValueError(f"Dimension reduction technique {DR} not supported")


    def __init_model(self):
        self.model = None

        if self.DR in ['shap', 'pca', 'sa']:
            indices = self.__select_important_features()
            self.domain = [self.domain[i] for i in indices]
            self.X = self.X.iloc[:, indices]
            cut_domain(indices) 
            # no need to cut y
        
        elif self.DR == 'rembo':
            self.projection = np.random.normal(size=(len(self.domain), self.n_relevant_features))
            pinv = np.linalg.pinv(self.projection)
            # print(self.X.shape)
            self.X = self.X.to_numpy() @ pinv.T
            # print(self.X.shape)

            lb = np.array([item['bounds'][0] for item in self.domain])
            ub = np.array([item['bounds'][1] for item in self.domain])
            set_input_transformer(lambda x: np.clip(x @ self.projection.copy().T, lb.copy(), ub.copy())) # x.T ?
        
        if self.AF == 'UCB':
            set_output_transformer(lambda x: -x)
            self.AF = 'LCB'

        is_rembo = False
        if self.DR == 'rembo':
            is_rembo = True
        else:
            self.X = self.X.to_numpy()
            self.y = self.y.to_numpy()

        if self.AF == 'DYCORS':
            # DYCORS-specific tweak
            y_rows = self.y.values.tolist() if isinstance(self.y, pd.DataFrame) else self.y.tolist()
            set_initial_design(y_rows)

            # init DYCORS
            self.model = init_DYCORS(self.domain, n_trials=self.n_iter, X=self.X, y=self.y, dim=self.n_relevant_features, is_rembo=is_rembo)
        else:
            # init GPyOpt BO
            self.model = init_BO(self.domain, self.X, self.y, acquisition_type=self.AF, dim=self.n_relevant_features, is_rembo=is_rembo)

    def run(self):
        if self.AF == 'DYCORS':
            self.model.run()
        else:
            self.model.run_optimization(max_iter=self.n_iter)
        pass