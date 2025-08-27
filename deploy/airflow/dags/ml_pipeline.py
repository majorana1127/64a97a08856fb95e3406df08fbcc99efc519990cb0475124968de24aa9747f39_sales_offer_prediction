from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import sys
sys.path.append('/opt/airflow/src')

import data_preprocessing
import model_training
import evaluation

# If you have these, otherwise make stubs:
try:
    import feature_engineering
except ImportError:
    feature_engineering = type('feature_engineering', (), {'feature_engineering': lambda: None})()

try:
    import drift_detection
except ImportError:
    drift_detection = type('drift_detection', (), {'detect_drift': lambda: {'drift_detected': False}})()

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
}

def branch_on_drift():
    import json
    try:
        with open("/opt/airflow/reports/drift_report.json") as f:
            data = json.load(f)
        if data.get("drift_detected"):
            return "retrain_model"
    except Exception:
        pass
    return "pipeline_complete"

with DAG('ml_pipeline',
         default_args=default_args,
         schedule_interval=None,
         catchup=False) as dag:

    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=data_preprocessing.preprocess_data
    )

    feature_eng_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering.feature_engineering  # stub if not implemented
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=model_training.train_model
    )

    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluation.evaluate_model
    )

    drift_task = PythonOperator(
        task_id='drift_detection',
        python_callable=lambda: drift_detection.detect_drift("/opt/airflow/data/test_X.csv", "/opt/airflow/data/drifted_test.csv")
    )

    branch_task = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=branch_on_drift
    )

    retrain_task = PythonOperator(
        task_id='retrain_model',
        python_callable=model_training.train_model
    )

    done_task = EmptyOpersator(task_id="pipeline_complete")

    # DAG structure
    preprocess_task >> feature_eng_task >> train_task >> evaluate_task >> drift_task >> branch_task
    branch_task >> retrain_task
    branch_task >> done_task