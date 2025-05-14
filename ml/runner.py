import subprocess
import sys
import os

def run_nohup_detached():
    log_path = os.path.join("..", "artefacts.log")
    log_file = open(log_path, "a")

    cmd = [
        "nohup",
        "python3.10",
        "main.py",
        "-a", "FILL_IN",
        "-d", "FILL_IN",
    ]

    popen_kwargs = {
        "stdout": log_file,
        "stderr": log_file,
        "stdin": subprocess.DEVNULL,
        "start_new_session": True,
    }

    subprocess.Popen(cmd, **popen_kwargs)
    log_file.close()

if __name__ == "__main__":
    run_nohup_detached()
