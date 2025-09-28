#!/usr/bin/env python3
"""
Toxic Terminator - Setup Check Utility
======================================

Verifies all dependencies and model files are correctly installed.

Usage: python setup_check.py
"""

import sys
import importlib
import os
from pathlib import Path

# Get the project root directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"


def check_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    try:
        importlib.import_module(import_name or package_name)
        return True
    except ImportError:
        return False


def main():
    """Perform all setup verification checks."""
    # Required packages: (display_name, import_name)
    required_packages = [
        ("streamlit", "streamlit"),
        ("nltk", "nltk"),
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
        ("numpy", "numpy")
    ]
    
    # Required NLTK resources (path, download_name)
    nltk_resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")
    ]
    
    print("🔍 Checking required packages...")
    missing_packages = []
    
    for display_name, import_name in required_packages:
        if check_package(display_name, import_name):
            print(f"  ✅ {display_name}")
        else:
            print(f"  ❌ {display_name}")
            missing_packages.append(display_name)
    
    print("\n📂 Checking model files...")
    model_files = ["tf_idf.pkt", "toxicity_model.pkt"]
    missing_files = []
    
    for file in model_files:
        file_path = MODELS_DIR / file
        if file_path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
            missing_files.append(file)
    
    print("\n📚 Checking NLTK resources...")
    import nltk
    
    for resource_path, resource_name in nltk_resources:
        try:
            nltk.data.find(resource_path)
            print(f"  ✅ {resource_name}")
        except LookupError:
            print(f"  ⬇️  Downloading {resource_name}...")
            nltk.download(resource_name, quiet=True)
            print(f"  ✅ {resource_name} (downloaded)")
    
    # Summary
    print("\n" + "=" * 40)
    if missing_packages:
        print("⚠️  Missing packages. Install with:")
        print(f"   pip install {' '.join(missing_packages)}")
    
    if missing_files:
        print("⚠️  Missing model files. Run the model.ipynb notebook first.")
    
    if not missing_packages and not missing_files:
        print("✅ All dependencies satisfied!")
        print("\n🚀 Run the app with:")
        print(f"   cd {SCRIPT_DIR}")
        print("   streamlit run app.py")


if __name__ == "__main__":
    main()