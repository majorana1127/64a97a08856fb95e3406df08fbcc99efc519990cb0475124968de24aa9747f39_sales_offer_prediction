# 64a97a08856fb95e3406df08fbcc99efc519990cb0475124968de24aa9747f39_sales_offer_prediction
# Sales Offer Prediction

This project predicts whether a potential customer will accept a deposit offer over a phone call. The dataset contains demographic and campaign-related features collected from previous marketing calls. The target is binary: yes or no to opening a deposit account. This was a dataset I had used previously when applying for work so I just reused the codes for my model then.

---

## Project Overview

This is a binary classification problem using a bank marketing dataset. each row represents a call made to a client, along with their personal and contact info, and whether they said yes or no to the offer.

we want to build a machine learning pipeline that:
- preprocesses the data
- trains a classifier
- evaluates performance
- and outputs predictions

In this case, the data did not need engineering but if it does, this will include a feature engineering step. The project is modular and production-ready, following the MLOps lifecycle.

---

## How to get the data

the dataset is stored in the `data/` directory as `bank.csv`.

if sharing this repo without data, it can be accessed in Kaggle: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset?resource=download

---

## Folder structure

project_root/

├── data/ - raw csv data

├── models/ -  saved xgboost model

├── outputs/ - predictions and logs

├── reports/ - evaluation metrics

├── src/ - all pipeline code

│ ├── data_preprocessing.py

│ ├── feature_engineering.py

│ ├── model_training.py

│ ├── evaluation.py

│ └── run_pipeline.py

├── tests/ - basic pytest script

├── .venv/ - uv virtual environment

├── requirements.txt

├── .pre-commit-config.yaml

└── README.md


i separated raw data, models, code, and results for clarity and reproducibility.

---

## Setup Instructions

this project uses [uv](https://astral.sh/blog/uv-intro/) for fast environment management.

### To run locally:

1. create a virtual environment: \uv\uv.exe venv
2. activate it: .venv\Scripts\activate
3. install dependencies: pip install pandas, numpy, scikit-learn, xgboost, pytest

## Running the pipeline

from the project root: python src/run_pipeline.py

this will:
- load and preprocess the data
- train an xgboost model with hyperparameter tuning
- evaluate accuracy, f1, and confusion matrix
- save metrics to `reports/metrics.txt`
- save predictions to `outputs/predictions.csv`
- save the model to `models/model.pkl`

## pre-commit configuration

this project uses pre-commit hooks to catch code issues early.

### hooks used:

- `ruff`: ensures consistent formatting and fast linting
- `trailing-whitespace`: removes trailing spaces
- `end-of-file-fixer`: adds newline at end of files

### Setup:
uv pip install pre-commit
pre-commit install
pre-commit run --all-files

this prevents commits unless the code passes all formatting checks.

---

## tests

basic unit tests are included in `tests/test_pipeline.py`.

they check that:
- data preprocessing returns valid splits
- model training returns a valid classifier
- evaluation creates the expected metrics output

to run tests: pytest

---

## Reflection

i'm used to working with notebooks, so it was a bit tricky adjusting to running everything through `.py` scripts and the command line. it took a while to get used to writing modular code and making sure all the files worked together through one pipeline script.

setting up the environment with uv was also confusing at first, especially on a windows work laptop where i didn’t have admin rights and couldn't add uv to my system path. i had to manually run uv from the downloads folder, which added friction to every step. even activating the virtual environment initially threw a script permission error, which i had to fix with execution policies.

pre-commit was another bump since ruff blocked my commits several times and i had to understand how linting worked, how to manually stage fixed files, and how to interpret error messages that weren’t always beginner-friendly.

it was challenging overall but i now appreciate the structure and discipline this kind of workflow enforces. it gave me a much clearer sense of what production-ready code actually looks like.
---
