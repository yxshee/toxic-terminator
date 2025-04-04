#!/usr/bin/env python3

import sys
import importlib
import subprocess
import os

def check_package(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def main():
    # Required packages
    required_packages = [
        "streamlit",
        "nltk",
        "sklearn",
        "pandas",
        "numpy"
    ]
    
    # NLTK resources
    nltk_resources = [
        "punkt",
        "wordnet",
        "averaged_perceptron_tagger",
        "stopwords"
    ]
    
    # Check for required packages
    print("Checking required packages...")
    missing_packages = []
    
    for package in required_packages:
        if check_package(package):
            print(f"✅ {package} is installed")
        else:
            print(f"❌ {package} is NOT installed")
            missing_packages.append(package)
    
    # Check for required model files
    print("\nChecking required model files...")
    model_files = ["toxicity_model.pkt", "tf_idf.pkt"]
    missing_files = []
    
    for file in model_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} does NOT exist")
            missing_files.append(file)
    
    # Check for NLTK resources
    print("\nChecking NLTK resources...")
    import nltk
    
    for resource in nltk_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}') if resource == 'punkt' else nltk.data.find(f'corpora/{resource}')
            print(f"✅ NLTK {resource} is downloaded")
        except LookupError:
            print(f"❌ NLTK {resource} is NOT downloaded")
            print(f"   Downloading NLTK {resource}...")
            nltk.download(resource)
    
    # Summary and fixes
    if missing_packages:
        print("\n⚠️ Some packages are missing. Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
    
    if missing_files:
        print("\n⚠️ Some model files are missing. Make sure the Toxic Terminator model files are in the current directory.")
    
    if not missing_packages and not missing_files:
        print("\n✅ All dependencies are satisfied. You can run the Toxic Terminator with:")
        print("streamlit run interface.py")

if __name__ == "__main__":
    main()