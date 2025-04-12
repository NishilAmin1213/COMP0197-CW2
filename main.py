#!/usr/bin/env python3
import sys
import subprocess

def run_script(script_path):
    try:
        print(f"Running {script_path}...")
        # Use sys.executable to run the script with the same interpreter.
        subprocess.run([sys.executable, script_path], check=True)
        print(f"{script_path} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_path}: {e}")
        sys.exit(1)

def main():
    # Run the weakly supervised model script.
    run_script("weakly_supervised_model.py")
    # Run the supervised model script.
    run_script("supervised_model.py")
    # Run the hybrid model script.
    run_script("oeq_hybrid_parallel_model.py")
    # Run the weakly supervised model script.
    run_script("oeq_weakly_and_supervised_model.py")


if __name__ == '__main__':
    main()
