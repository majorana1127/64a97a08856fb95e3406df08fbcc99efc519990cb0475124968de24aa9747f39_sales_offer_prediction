# 64a97a08856fb95e3406df08fbcc99efc519990cb0475124968de24aa9747f39_sales_offer_prediction

# Sales Offer Prediction

This project predicts whether a potential customer will accept a deposit offer over a phone call. The dataset contains demographic and campaign-related features collected from previous marketing calls. The target is binary: yes or no to opening a deposit account. This was a dataset I had used previously when applying for work, so I just reused the code for my model then.

---

## Project Overview

This is a binary classification problem using a bank marketing dataset. Each row represents a call made to a client, along with their personal and contact info, and whether they said yes or no to the offer.

This pipeline:
- Preprocesses the data
- Trains an XGBoost classifier with hyperparameter tuning
- Evaluates model performance
- Outputs predictions, metrics, and the trained model
- Is orchestrated using Apache Airflow and runs inside Docker containers

The project is modular and production-ready, following the MLOps lifecycle.

---

## How to Get the Data

The dataset is stored in the `data/` directory as `bank.csv`.

If sharing this repo without data, it can be accessed from Kaggle:
https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset?resource=download

---

## Folder Structure

project_root/
├── data/ # Raw CSV data
├── models/ # Saved XGBoost model
├── outputs/ # Predictions
├── reports/ # Evaluation metrics
├── src/ # All pipeline code
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── model_training.py
│ ├── evaluation.py
│ └── run_pipeline.py
├── tests/ # Unit tests
├── deploy/
│ └── airflow/ # Airflow DAGs and Docker setup
│ ├── dags/
│ │ └── ml_pipeline.py
│ ├── docker-compose.yml
│ └── Dockerfile
├── .pre-commit-config.yaml
├── requirements.txt
└── README.md

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

## Docker and Airflow Setup

to run with Airflow in Docker:

1. Go to the Airflow directory:
    ```
    cd deploy/airflow
    ```

2. Build and start containers:
    ```
    docker-compose up -d --build
    ```

3. Access the Airflow UI:
    ```
    http://localhost:8080
    ```
    Login with:
    - Username: `airflow`
    - Password: `airflow`

4. Trigger the DAG `ml_pipeline` manually from the UI or test individual tasks using the command line (see below).

## Task Testing (Airflow CLI):

To test individual DAG tasks without triggering the full DAG:
```
docker-compose exec webserver airflow tasks test ml_pipeline preprocess 2025-08-01
docker-compose exec webserver airflow tasks test ml_pipeline train 2025-08-01
docker-compose exec webserver airflow tasks test ml_pipeline evaluate 2025-08-01
```
Logs will be saved under `airflow/logs/`.

## pre-commit configuration

this project uses pre-commit hooks to catch code issues early.

### hooks used:

- `ruff`: Python formatting and linting
- `trailing-whitespace`: Removes trailing spaces
- `end-of-file-fixer`: Ensures newline at EOF
- `yamllint`: Validates YAML files (e.g., docker-compose.yml)

### Setup:
uv pip install pre-commit
pre-commit install
pre-commit run --all-files


### Justification:

This project uses yamllint for validating YAML files such as docker-compose.yml. yamllint helps ensure proper formatting and structure, which is critical for Docker orchestration. hadolint was not included to avoid additional system dependencies on Windows, but yamllint sufficiently covers config validation for this project.
---

## Tests

Basic unit tests are included in `tests/test_pipeline.py`.

They check that:
- Data preprocessing returns valid splits
- Model training returns a valid classifier
- Evaluation creates the expected metrics output

To run tests: pytest

---

## Reflection HW1

I'm used to working with notebooks, so it was a bit tricky adjusting to running everything through `.py` scripts and the command line. It took a while to get used to writing modular code and making sure all the files worked together through one pipeline script.

Setting up the environment with uv was also confusing at first, especially on a Windows work laptop where I didn’t have admin rights and couldn't add uv to my system path. I had to manually run uv from the downloads folder, which added friction to every step. Even activating the virtual environment initially threw a script permission error, which I had to fix with execution policies.

Pre-commit was another bump since ruff blocked my commits several times and I had to understand how linting worked, how to manually stage fixed files, and how to interpret error messages that weren’t always beginner-friendly.

It was challenging overall but I now appreciate the structure and discipline this kind of workflow enforces. It gave me a much clearer sense of what production-ready code actually looks like.

---

## Reflection HW2

I had no prior experience with Docker or Airflow, so even just understanding what went where felt overwhelming. The setup instructions seemed simple at first, but in reality, every small thing like missing files, unrecognized commands, or misconfigured paths turned into hours of debugging.

I ran into issues like missing poetry.lock, failed Docker builds, ports not working, and modules not being found inside containers even though they worked locally. At one point, I couldn’t even get localhost:8080 to load, and when it finally did, the Airflow login didn’t work. There were times it felt like everything was broken and I wanted to give up.

Even so, I learned how to read logs properly, how to debug DAGs, how to fix broken containers, and how to use pre-commit for Docker/YAML checks. More importantly, I got a working, orchestrated ML pipeline that runs inside containers — something that sounded impossible a few days ago.
