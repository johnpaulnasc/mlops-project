# pipeline/airflow/churn_pipeline.py
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from src.data_preprocessing import preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 9, 17),
}

def preprocessing():
    preprocess_data('data/raw_data.csv', 'data/')

def train():
    train_model('data/train_data.csv')

def evaluate():
    evaluate_model('data/test_data.csv', 'models/')

with DAG(dag_id='churn_pipeline', default_args=default_args, schedule_interval='@daily') as dag:
    preprocess_task = PythonOperator(task_id='preprocess_data', python_callable=preprocessing)
    train_task = PythonOperator(task_id='train_model', python_callable=train)
    evaluate_task = PythonOperator(task_id='evaluate_model', python_callable=evaluate)

    preprocess_task >> train_task >> evaluate_task