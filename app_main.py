# import subprocess

# # Global variables to hold references to subprocesses
# main_process = None
# app_process = None

# def run_main_and_app():
#     global main_process, app_process
#     main_process = subprocess.Popen(['python', 'main.py'])
#     app_process = subprocess.Popen(['python', 'app.py'])

# def terminate_subprocesses():
#     global main_process, app_process
#     if main_process and main_process.poll() is None:
#         main_process.terminate()
#     if app_process and app_process.poll() is None:
#         app_process.terminate()

# if __name__ == '__main__':
#     try:
#         # Run main.py and app.py together
#         run_main_and_app()

#         # Wait for Ctrl+C
#         while True:
#             pass

#     except KeyboardInterrupt:
#         print("\nCtrl+C pressed. Shutting down...")
#         terminate_subprocesses()

#     print("Launcher script completed.")

# import multiprocessing
# import os
# import signal
# import time

# def run_app(file_name):
#     os.system(f"python {file_name}")

# def stop_processes(processes):
#     for proc in processes:
#         if proc.is_alive():
#             os.kill(proc.pid, signal.SIGTERM)  # Send termination signal

# if __name__ == "__main__":
#     processes = []
#     files = ['main.py', 'app.py']

#     try:
#         for file in files:
#             proc = multiprocessing.Process(target=run_app, args=(file,))
#             proc.start()
#             processes.append(proc)

#         while True:  # Keep the script running
#             time.sleep(1)

#     except KeyboardInterrupt:
#         print("Stopping applications...")
#         stop_processes(processes)
# ************************************************************

import os
import subprocess
import sys
import time
import signal

LOCK_FILE = "app.lock"
processes = []

def acquire_lock():
    """Create a lock file to prevent multiple instances."""
    try:
        with open(LOCK_FILE, "x"):
            pass
    except FileExistsError:
        print("Another instance is already running.")
        sys.exit(1)

def release_lock():
    """Remove the lock file."""
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

def run_app(file_name):
    """Start subprocess in the same process group so Ctrl+C works."""
    proc = subprocess.Popen([sys.executable, file_name])
    processes.append(proc)

def stop_processes(signum=None, frame=None):
    """Terminate all subprocesses."""
    print("\nStopping applications...")
    for proc in processes:
        try:
            proc.terminate()  # send SIGTERM
        except Exception:
            pass
    release_lock()
    sys.exit(0)

if __name__ == "__main__":
    acquire_lock()

    # Register signal handlers for Ctrl+C and termination
    signal.signal(signal.SIGINT, stop_processes)
    signal.signal(signal.SIGTERM, stop_processes)

    # Files to run
    files = ['main.py', 'app.py']

    try:
        # Start each subprocess
        for file in files:
            run_app(file)

        # Keep launcher alive
        while True:
            time.sleep(0.1)

    finally:
        release_lock()
