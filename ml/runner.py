import subprocess
import sys
import os

def run_nohup_detached():
    # Path to your log (relative to cwd or absolute)
    log_path = os.path.join("..", "artefacts.log")
    # Open the log file in append mode
    log_file = open(log_path, "a")

    # Build the full command: nohup python3.10 main.py -m BO
    cmd = [
        "nohup",
        "python3.10",  # or hardcode "python3.10"
        "main.py",
        "-m", "BO"
    ]

    # On POSIX, start_new_session detaches the child (like setsid())
    popen_kwargs = {
        "stdout": log_file,
        "stderr": log_file,
        "stdin": subprocess.DEVNULL,
        "start_new_session": True,
    }

    # Launch
    subprocess.Popen(cmd, **popen_kwargs)

    # Close our handle; child keeps writing
    log_file.close()

if __name__ == "__main__":
    run_nohup_detached()
