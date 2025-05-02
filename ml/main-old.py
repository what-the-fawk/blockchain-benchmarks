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

# 6. Sensitivity Analysis Feature Importance
def sensitivity_analysis_feature_importance(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)

    problem = {
        'num_vars': X.shape[1],
        'names': X.columns.tolist(),
        'bounds': [[X[col].min(), X[col].max()] for col in X.columns]
    }

    sp = ProblemSpec(problem)
    sp.sample_saltelli(1000)

    sp.set_results(model.predict(sp.samples))
    sp.analyze_sobol()

    sobol_df = pd.DataFrame({
        'Feature': problem['names'],
        'S1': sp.analysis['S1'],
        'ST': sp.analysis['ST']
    }).sort_values(by='ST', ascending=False)

    return sobol_df

def pca_feature_importance(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X, y)

    loadings = pd.DataFrame(
            np.abs(pca.components_),
            columns=X.columns
        )

    explained_variance = pca.explained_variance_ratio_
    weighted_loadings = loadings.multiply(explained_variance, axis=0)
    importance_scores = weighted_loadings.sum(axis=0).sort_values(ascending=False)

    results = pd.DataFrame({
        'Feature': importance_scores.index,
        'Importance': importance_scores.values
    }).reset_index(drop=True)

    return results

def read_extract_config(config):
    number_with_unit_pattern = re.compile(r'^\s*(-?\d+(\.\d+)?)(.*)$')

    extracted_data = []

    def traverse(node, path=''):
        if isinstance(node, dict):
            for key, value in node.items():
                new_path = f"{path}.{key}" if path else key
                traverse(value, new_path)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                new_path = f"{path}[{index}]"
                traverse(item, new_path)
        elif isinstance(node, str):
            match = number_with_unit_pattern.match(node)
            if match:
                number = float(match.group(1))
                unit = match.group(3).strip()
                full_path = f"{path}|{unit}" if unit else path  # Append unit if available
                extracted_data.append({'path': full_path, 'value': number})
        elif isinstance(node, (int, float)):
            extracted_data.append({'path': path, 'value': node})

    traverse(config)
    return extracted_data

def load_config(path: str):
    """
        Load the configuration file
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def transform_caliper_html_to_json(html_file, output_json_file):
    """
        Transform html report file to json format
    """
    try:
        with open(html_file, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
        
        summary = soup.find("div", {"id": "summary"}).text.strip() if soup.find("div", {"id": "summary"}) else "No summary found"
        metrics = {}
        
        table = soup.find("table")
        if table:
            headers = [th.text.strip() for th in table.find_all("th")]
            rows = table.find_all("tr")[1:]
            for row in rows:
                cells = row.find_all("td")
                key = cells[0].text.strip()
                values = {headers[i]: cells[i].text.strip() for i in range(1, len(cells))}
                metrics[key] = values
        
        report_data = {
            "summary": summary,
            "metrics": metrics
        }
        
        with open(output_json_file, "w", encoding="utf-8") as json_file:
            json.dump(report_data, json_file, indent=4)
        
        print(f"JSON report successfully created: {output_json_file}")
    except Exception as e:
        print(f"Error during conversion: {e}")


def calculate_average_tps(json_file):
    try:
        json_data = None
        with open(json_file, "r", encoding="utf-8") as file:
            json_data = json.load(file)
        tps_values = []
        for _, metrics in json_data["metrics"].items():
            tps_value = float(metrics.get("Throughput (TPS)", 0))
            tps_values.append(tps_value)
        
        if not tps_values:
            print("No TPS values found.")
            return None
        
        average_tps = sum(tps_values) / len(tps_values)
        return average_tps

    except Exception as e:
        print(f"Error: {e}")
        return None


def objective_function():
    # use core.yaml
    subprocess.run('export FABRIC_CFG_PATH=core.yaml', shell=True)

    subprocess.run('cd ../fabric-samples/test-network && ./network.sh up createChannel -s couchdb', shell=True)

    #deploy chaincode
    chaincode_deploy = 'cd ../fabric-samples/test-network && ./network.sh deployCC -ccn fabcar -ccp ../../caliper-benchmarks/src/fabric/samples/fabcar/go -ccl go'
    subprocess.run(chaincode_deploy, shell=True)

    # execute benchmark
    bencmmark = "cd ../caliper-benchmarks/ && npx caliper launch manager --caliper-workspace ./ --caliper-networkconfig networks/fabric/test-network.yaml --caliper-benchconfig benchmarks/samples/fabric/fabcar/config.yaml --caliper-flow-only-test --caliper-fabric-gateway-enabled"
    subprocess.run(bencmmark, shell=True)

    # process output
    transform_caliper_html_to_json('../caliper-benchmarks/report.html', 'report.json')
    objective = calculate_average_tps('report.json')

    # shut down network
    subprocess.run('cd ../fabric-samples/test-network && ./network.sh down', shell=True)
    return objective


def fill_config(conf, next_point, sample_params):
    for i in range(len(sample_params)):
        try:
            internal_path = sample_params[i].name.split('.')
            parts = internal_path[-1].split('|')
            assert(len(parts) > 0 and len(parts) <= 2)
            internal_path[-1] = parts[0]

            assert(len(internal_path) > 0)
            link = conf[internal_path[0]]
            for j in range(1, len(internal_path)):
                link = link[internal_path[j]]
            # print(link)
            if len(parts) == 1:
                link = next_point[i]
            else:
                link = str(int(next_point[i])) + parts[1]
            pass
        except Exception as e:
            print(f"Error: {e}")
            continue


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    CONFIG_PATH = '../fabric-samples/test-network/configtx/configtx.yaml'
    conf = load_config(str(CONFIG_PATH))
    conf_core = load_config('core.yaml')

    # assert(False, "Config path")
    # logger.info(f"Loaded configuration from {CONFIG_PATH}")

    sample_params = [
        Integer(1, 19, name="Orderer.BatchSize.MaxMessageCount|"), # follow style of conf
        Integer(1, 49, name="Orderer.BatchSize.AbsoluteMaxBytes|MB"), # 49 is max recommended
        Integer(1, 2048, name="Orderer.BatchSize.PreferredMaxBytes|KB"),
        Integer(1, 30, name="Orderer.BatchTimeout|s"),
        Integer(1, 999, name="Profiles.ChannelUsingRaft.Orderer.EtcdRaft.Options.TickInterval|ms"),
        Integer(1, 100, name="Profiles.ChannelUsingRaft.Orderer.EtcdRaft.Options.ElectionTick"),
        Integer(1, 10, name="Profiles.ChannelUsingRaft.Orderer.EtcdRaft.Options.HeartbeatTick"),
        Integer(1, 50, name="Profiles.ChannelUsingRaft.Orderer.EtcdRaft.Options.MaxInflightBlocks"),
        Integer(1, 128, name="Profiles.ChannelUsingRaft.Orderer.EtcdRaft.Options.SnapshotIntervalSize|MB"),
    ]

    # print(conf_core)
    core_data = read_extract_config(conf_core)
    # print(core_data)

    for mapping in core_data:
        name = mapping['path']
        if "address" in name.lower():
            continue
        value = mapping['value']
        print(name, value)
        if isinstance(value, float):
            sample_params.append(Real(value * 0.75, value * 1.5, name=name))
        elif isinstance(value, int):
            sample_params.append(Integer(value - 10, value + 10, name=name))

    print(len(sample_params))
    # print(sample_params, sep="\n")
    # raise RuntimeError("Stop here")

    # TODO: orderer.type == BFT

    NUM_EPOCHS = 2
    NUM_SNR = 1
    # learning
    # TODO: divide all values by the first one (normalize)

    optimizer = Optimizer(
        dimensions=sample_params,
        base_estimator="GP",
        n_initial_points=10,
        acq_func="gp_hedge",
        acq_optimizer="auto",
    )

    def fit(sample_params, optimizer, conf, scores_snr_path="scores_snr.txt", optimals_path="optimals.txt", scores_path="scores.txt") -> tuple[pd.DataFrame, pd.Series]:

        # cleanup
        if os.path.exists(scores_path):
            os.remove(scores_path)
        if os.path.exists(optimals_path):
            os.remove(optimals_path)
        if os.path.exists(scores_snr_path):
            os.remove(scores_snr_path)

        scores = []
        optimals = []
        scores_snr = []

        df = pd.DataFrame(columns=[param.name.split('|')[0] for param in sample_params])
        y = pd.Series(name="Throughput (TPS)")
        # TODO: take failed transactions into account

        for i in range(NUM_EPOCHS):
            next_point = optimizer.ask()

            df.loc[len(df)] = next_point

            if next_point is None:
                print("No more points to evaluate.")
                break

            fill_config(conf, next_point, sample_params)
            fill_config(conf_core, next_point, sample_params) # should throw when given params for other configs

            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(conf, f)

            with open('core.yaml', 'w') as f:
                yaml.dump(conf_core, f)

            print(f"Next point to evaluate: {next_point}")
            optimals.append(next_point)

            snr_scores = []
            for _ in range(NUM_SNR):
                value = objective_function()
                snr_scores.append(value)
                pass

            scores_snr.append(snr_scores)
            scores.append(snr_scores[0])
            y.loc[len(y)] = snr_scores[0]
            print(f"Score: {value}\n")
            optimizer.tell(next_point, -value)

            with open(scores_path, "a") as f:
                print(scores[-1], file=f)
            
            with open(optimals_path, "a") as f:
                print(optimals[-1], file=f)

            with open(scores_snr_path, "a") as f:
                print(scores_snr[-1], file=f)

        return df, y

    profiler = LineProfiler()
    profiler.add_function(fit)
    profiler.enable()
    df, y = fit(sample_params, optimizer, conf)
    profiler.disable()
    with open("line_profiler_stats.txt", "w") as f:
        profiler.print_stats(stream=f)

    shap_importance = shap_feature_importance(X=df, y=y)
    rf_importance = random_forest_feature_importance(X=df, y=y)
    lasso_importance = lasso_feature_importance(X=df, y=y)
    perm_importance = permutation_feature_importance(X=df, y=y)
    gb_importance = gradient_boosting_feature_importance(X=df, y=y)
    sa_feature_importance = sensitivity_analysis_feature_importance(X=df, y=y)
    pca_importance = pca_feature_importance(X=df, y=y)

    shap_importance.to_csv('shap_feature_importance.csv', index=False)
    rf_importance.to_csv('random_forest_feature_importance.csv', index=False)
    lasso_importance.to_csv('lasso_feature_importance.csv', index=False)
    perm_importance.to_csv('permutation_feature_importance.csv', index=False)
    gb_importance.to_csv('gradient_boosting_feature_importance.csv', index=False)
    sa_feature_importance.to_csv('sensitivity_analysis_feature_importance.csv', index=False)
    pca_importance.to_csv('pca_feature_importance.csv', index=False)

    n = 10
    filtered_samples = []

    for importance_df in [
        shap_importance, rf_importance, lasso_importance, 
        perm_importance, gb_importance, sa_feature_importance, pca_importance
    ]:
        top_features = set(importance_df['Feature'].head(n))
        
        filtered_sample_params = [
            param for param in sample_params if param.name.split('|')[0] in top_features
        ]
        filtered_samples.append(filtered_sample_params)

    optimizer = Optimizer(
        dimensions=sample_params,
        base_estimator="GP",
        n_initial_points=10,
        acq_func="gp_hedge",
        acq_optimizer="auto",
    )

    experiment_count = "1"
    
    for params in filtered_samples:
        fit(params, optimizer, conf, scores_snr_path=experiment_count + "_scores_snr.txt", optimals_path=experiment_count + "_optimals.txt", scores_path=experiment_count + "_scores.txt")
        experiment_count = str(int(experiment_count) + 1)




if __name__ == "__main__":
    main()