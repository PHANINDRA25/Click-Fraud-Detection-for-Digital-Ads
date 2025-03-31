# Databricks notebook source
# MAGIC %md
# MAGIC ## Creating project structure programmatically - different folders and files within it.

# COMMAND ----------

import os

# Folders to create
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "models",
    "mlruns",
    "tests",
    ".github/workflows",
    "config"
]

# Files to create with placeholder content
files = {
    ".gitignore": """
__pycache__/
*.pyc
*.pyo
*.pyd
env/
venv/
build/
dist/
.eggs/
*.egg-info/
.ipynb_checkpoints
.vscode/
mlruns/
.DS_Store
""",

    "README.md": "# Production ML Pipeline with LightGBM, MLflow, and CI/CD\n",

    "requirements.txt": """
lightgbm
mlflow
scikit-learn
pandas
pyyaml
pytest
""",

    ".github/workflows/ml_pipeline.yaml": """
name: ML Pipeline CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test-train:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run unit tests
      run: |
        pytest tests/

    - name: Run training script
      run: |
        python notebooks/02_train_lightgbm.py
""",

    "src/logger.py": """
import logging
import sys

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
""",

    "src/config.py": """
import yaml

def load_config(path='config/model_config.yaml'):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config
""",

    "config/model_config.yaml": """
params:
  objective: binary
  metric: auc
  learning_rate: 0.1
  num_leaves: 31
  verbosity: -1
  random_state: 42
""",

    "config/environment.yaml": """
name: ml-env
channels:
  - defaults
dependencies:
  - python=3.9
  - pip
  - pip:
      - -r requirements.txt
""",

    "notebooks/01_data_preprocessing.py": "# Placeholder: Data preprocessing code\n",
    "notebooks/02_train_lightgbm.py": "# Placeholder: Training LightGBM model + MLflow logging\n",
    "notebooks/03_batch_inference.py": "# Placeholder: Batch inference code\n",

    "src/data_utils.py": "# Placeholder: load_data(), clean_data(), etc.\n",
    "src/model_utils.py": "# Placeholder: train_model(), save_model(), etc.\n",
    "src/eval_utils.py": "# Placeholder: evaluate_model(), plot_roc_curve(), etc.\n",

    "tests/test_data_utils.py": "def test_dummy():\n    assert True\n",
    "tests/test_model_utils.py": "def test_dummy():\n    assert True\n"
}


def create_structure():
    # Create folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Created folder: {folder}")

    # Create files
    for path, content in files.items():
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(path, "w") as f:
            f.write(content.strip() + "\n")
        print(f"📄 Created file: {path}")


if __name__ == "__main__":
    create_structure()
