import yaml
import subprocess
import sys

class NetworkStatus:
    pass

def set_envs(yaml) -> dict:
    """
    Set the environment variables from the yaml file
    """
    envs = {}
    for key, value in yaml.items():
        if isinstance(value, dict):
            envs.update(set_envs(value))
        else:
            envs[key] = value
    return envs


network_containers = None

# def set_containers(config: str):
#     network_containers = ""

def start_network(config: str)-> NetworkStatus:
    # Load the network configuration
    raise RuntimeError("TODO")
    if network_containers is not None:
        raise RuntimeError("Network is already running")

    with open(config) as f:
        network_config = yaml.load(f, Loader=yaml.SafeLoader)

    subprocess.run(["./network.sh", "start"], check=True, env=set_envs(network_config))

    return NetworkStatus()

def setup_files(config: str):
    raise RuntimeError("TODO setup function")
    pass

def stop_network()-> NetworkStatus:
    if network_containers is None:
        raise RuntimeError("Network is not running")
    
    subprocess.run(["./network.sh", "stop"], check=True, env=network_containers)
    return NetworkStatus() # ...

def main():

    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python network_starter.py <config_file_path>")

    cfg_path = sys.argv[1]
    with open(cfg_path, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    # setup files for network
    raise RuntimeError("TODO setup")

    start_network(cfg)





    pass