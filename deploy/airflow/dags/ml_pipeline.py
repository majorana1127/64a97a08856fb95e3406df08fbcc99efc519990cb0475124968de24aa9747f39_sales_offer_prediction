from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
sys.path.append('/opt/airflow/src')

import data_preprocessing
import model_training
import evaluation

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
}

with DAG('ml_pipeline',
         default_args=default_args,
         schedule_interval=None,
         catchup=False) as dag:

    preprocess_task = PythonOperator(
        task_id='preprocess',
        python_callable=data_preprocessing.preprocess_data
    )

    train_task = PythonOperator(
        task_id='train',
        python_callable=model_training.train_model
    )

    evaluate_task = PythonOperator(
        task_id='evaluate',
        python_callable=evaluation.evaluate_model
    )

    preprocess_task >> train_task >> evaluate_task
