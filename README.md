# 64a97a08856fb95e3406df08fbcc99efc519990cb0475124968de24aa9747f39_sales_offer_prediction

# Sales Offer Prediction

This project predicts whether a potential customer will accept a deposit offer over a phone call. The dataset contains demographic and campaign-related features collected from previous marketing calls. The target is binary: yes or no to opening a deposit account. The dataset is from Kaggle, and I originally used it for a prior job application.

---

## Project Overview

This is a binary classification pipeline using a bank marketing dataset. Each row represents a call made to a client, along with their personal and contact info, and whether they said yes or no to the offer.

### HW3 Pipeline Features

- Preprocesses the data (including simulated data drift for testing)
- Trains an XGBoost classifier (MLflow tracks parameters/artifacts)
- Evaluates model performance (metrics logged to MLflow)
- Detects drift with [Evidently](https://evidentlyai.com/) (where environment allows)
- Orchestrates all steps in Airflow (branches if drift detected)
- Runs end-to-end inside Docker containers

This project is modular and follows MLOps practices.

---

## HW3 New Features & Challenges


- **MLflow integration:** All training, evaluation, and pipeline steps are logged to an MLflow tracking server running in Docker.
- **Custom PyFunc model:** The pipeline saves the trained XGBoost model with a custom MLflow wrapper.
- **Drift simulation and detection:** Preprocessing generates drifted test/train data and a drift detector (Evidently) checks for data drift.
- **Airflow DAG branching:** The DAG branches to retrain or complete the pipeline based on drift detection results.

### Major Technical Challenges

> #### Python/Windows/Evidently Compatibility
> Setting up `evidently` for drift detection proved extremely difficult on Windows with Python 3.11+. Despite several clean venvs, forced reinstalls, and package downgrades, the module import failed due to pip/venv path issues and (likely) version conflicts. This issue persisted even when other packages (mlflow, scikit-learn, xgboost, etc.) worked fine in the same environment.
>
> As a result, the drift detection code is included and works in principle, but could not be demonstrated end-to-end on my local system.
---

## How to Get the Data

The dataset is stored in the `data/` directory as `bank.csv`.

If sharing this repo without data, it can be accessed from Kaggle:  
https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset?resource=download

---

## Folder Structure

```
project_root/
├── data/                 # Raw and processed data
├── models/               # Trained model(s)
├── outputs/              # Predictions
├── reports/              # Evaluation metrics and drift reports
├── src/                  # All pipeline code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── drift_detection.py
│   └── run_pipeline.py
├── tests/                # Unit tests
├── deploy/
│   └── airflow/
│       ├── dags/
│       │   └── ml_pipeline.py
│       ├── docker-compose.yml
│       └── Dockerfile
├── .pre-commit-config.yaml
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Environment

This project uses Python 3.11 and [uv](https://astral.sh/blog/uv-intro/) for fast environment management (but works with pip too).

**To run locally:**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # (Windows PowerShell)
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Pipeline

From the project root:
```bash
python -m src.run_pipeline
```

This will:
- Download and preprocess the data if missing
- Train an XGBoost model (logs run to MLflow)
- Evaluate accuracy and f1-score (logs metrics to MLflow)
- (Attempt) to run drift detection (skipped if environment blocks Evidently)
- Save metrics to `reports/evaluation_results.json`
- Save predictions to `outputs/predictions.csv`
- Save the model to `models/model.pkl`

---

## MLflow and Airflow Setup (HW3)

MLflow and Airflow run via Docker Compose for easy reproducibility:

1. **Copy `docker-compose.yml` to the project root** (or use `-f` flag from `deploy/airflow`).
2. **Start the containers:**
    ```bash
    docker compose up -d
    ```
3. **Access:**
    - MLflow UI: [http://localhost:5000](http://localhost:5000)
    - Airflow UI: [http://localhost:8080](http://localhost:8080)
      - Username: `airflow`
      - Password: `airflow`
4. **Trigger the DAG** `ml_pipeline` manually from the Airflow UI or test tasks individually (see below).

---

## Airflow Task Testing (CLI)

To test individual tasks:
```bash
docker compose exec airflow airflow tasks test ml_pipeline preprocess_data 2025-08-01
docker compose exec airflow airflow tasks test ml_pipeline train_model 2025-08-01
docker compose exec airflow airflow tasks test ml_pipeline evaluate_model 2025-08-01
docker compose exec airflow airflow tasks test ml_pipeline drift_detection 2025-08-01
```

---

## pre-commit configuration

This project uses pre-commit hooks to catch code issues early.

**Hooks used:**
- `ruff`: Python formatting and linting
- `trailing-whitespace`: Removes trailing spaces
- `end-of-file-fixer`: Ensures newline at EOF
- `yamllint`: Validates YAML files (e.g., docker-compose.yml)

**Setup:**
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
*Yamllint is used for Docker configs. Hadolint not included to avoid Windows system dependency headaches.*

---

## Tests

Basic unit tests are included in `tests/`.

They check:
- Data preprocessing returns valid splits
- Model training returns a valid classifier
- Evaluation creates the expected metrics output

**To run tests:**
```bash
pytest
```

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

---

## Reflection HW3: MLOps Integration & Drift Detection

This homework was the most technically challenging for me so far, mostly due to compatibility and environment issues with new libraries on Windows. MLflow and Docker were straightforward after HW2, but integrating Evidently for drift detection proved unexpectedly difficult. 

I spent hours trying different versions, recreating virtual environments, and even switching Python interpreters. The `evidently` package would install but never import successfully, always throwing `ModuleNotFoundError` for submodules (even when `pip show evidently` showed the right version and path). Using direct file paths, absolute pip/python calls, and even manual copying of the library did not solve the problem.

Because of this, the drift detection step is implemented in the codebase but could not be run to completion on my laptop.

---