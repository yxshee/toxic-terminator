#!/usr/bin/env python3
"""
Launcher script for the Toxic Terminator application.
This script provides a convenient way to run the Streamlit app 
without having to remember the Streamlit CLI command.
"""

import os
import sys
import subprocess

def main():
    """
    Main function to launch the Streamlit app.

    This function locates the app.py file and runs it using the Streamlit CLI.
    """
    # Get the absolute path to the app.py file
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    print(f"Starting Toxic Terminator app from: {app_path}")
    print("Launching with Streamlit...")
    
    # Run streamlit as a subprocess
    try:
        # Using subprocess.run to execute streamlit command
        result = subprocess.run(
            ["streamlit", "run", app_path],
            check=True,
        )
        return result.returncode
    except FileNotFoundError:
        print("ERROR: 'streamlit' command not found. Is Streamlit installed?")
        print("Install with: pip install streamlit")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Streamlit process failed with return code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nShutting down Toxic Terminator app...")
        return 0

if __name__ == "__main__":
    sys.exit(main())
