from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook 

from datetime import datetime, timedelta
import pandas as pd
import requests
import numpy as np
import os
import holidays
import pickle
import shutil
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- КОНФИГУРАЦИЯ ---
DATA_PATH = '/opt/airflow/dags/data/taxi_data.csv'
PROCESSED_PATH = '/tmp/merged_nyc_data.csv'
BEST_MODEL_PATH = '/opt/airflow/dags/data/best_price_model.pkl'
BUCKET_NAME = 'taxi-price-prediction-data' 

# --- НАСТРОЙКИ TELEGRAM (Вставь свои данные!) ---
TG_BOT_TOKEN = os.getenv('TG_TOKEN')
TG_CHAT_ID = os.getenv('TG_CHAT_ID')

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# --- ФУНКЦИИ ---

def send_telegram_message(text):
    """Отправляет сообщение в Telegram"""
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    params = {'chat_id': TG_CHAT_ID, 'text': text}
    try:
        requests.get(url, params=params)
    except Exception as e:
        print(f"Ошибка отправки в Telegram: {e}")

def extract_transform_taxi(**kwargs):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Файл {DATA_PATH} не найден.")
    
    cols_to_use = ['tpep_pickup_datetime', 'trip_distance', 'total_amount']
    df = pd.read_csv(DATA_PATH, usecols=cols_to_use, nrows=50000)
    df.columns = ['pickup_datetime', 'distance', 'price']
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['date_hour'] = df['pickup_datetime'].dt.floor('H')
    
    df = df[(df['price'] > 0) & (df['price'] < 300)]
    df = df[(df['distance'] > 0) & (df['distance'] < 100)]
    
    us_holidays = holidays.US()
    df['is_holiday'] = df['pickup_datetime'].dt.date.apply(lambda x: 1 if x in us_holidays else 0)
    
    df.to_parquet('/tmp/taxi_clean.parquet')
    
    min_date = df['date_hour'].min().strftime('%Y-%m-%d')
    max_date = df['date_hour'].max().strftime('%Y-%m-%d')
    return {'min_date': min_date, 'max_date': max_date}

def extract_weather_api(**kwargs):
    ti = kwargs['ti']
    dates = ti.xcom_pull(task_ids='extract_taxi_data')
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 40.71,
        "longitude": -74.01,
        "start_date": dates['min_date'],
        "end_date": dates['max_date'],
        "hourly": "temperature_2m,precipitation,rain,wind_speed_10m",
        "timezone": "America/New_York"
    }
    r = requests.get(url, params=params)
    data = r.json()
    hourly_data = data['hourly']
    
    df_weather = pd.DataFrame({
        'date_hour': hourly_data['time'],
        'temperature': hourly_data['temperature_2m'],
        'precipitation': hourly_data['precipitation'],
        'wind_speed': hourly_data['wind_speed_10m']
    })
    df_weather['date_hour'] = pd.to_datetime(df_weather['date_hour'])
    df_weather.to_parquet('/tmp/weather.parquet')

def merge_and_prepare(**kwargs):
    df_taxi = pd.read_parquet('/tmp/taxi_clean.parquet')
    df_weather = pd.read_parquet('/tmp/weather.parquet')
    merged = pd.merge(df_taxi, df_weather, on='date_hour', how='inner')
    merged.dropna(inplace=True)
    merged.to_csv(PROCESSED_PATH, index=False)

def load_to_postgres(**kwargs):
    df = pd.read_csv(PROCESSED_PATH)
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = pg_hook.get_sqlalchemy_engine()
    df.to_sql('nyc_taxi_weather', engine, if_exists='replace', index=False)

def upload_to_s3(**kwargs):
    execution_date = kwargs['ds'] 
    s3_key = f"processed_data_{execution_date}.csv"
    hook = S3Hook(aws_conn_id='aws_s3_conn')
    hook.load_file(filename=PROCESSED_PATH, key=s3_key, bucket_name=BUCKET_NAME, replace=True)

def check_data_quality(**kwargs):
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    records = pg_hook.get_first("SELECT COUNT(*) FROM nyc_taxi_weather")
    count = records[0]
    if count > 1000:
        return ['train_rf', 'train_gb', 'train_ridge', 'train_tree']
    else:
        return 'stop_pipeline'

def train_model(model_type, **kwargs):
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    df = pg_hook.get_pandas_df("SELECT * FROM nyc_taxi_weather")
    features = ['distance', 'temperature', 'precipitation', 'wind_speed', 'is_holiday']
    X = df[features]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor(max_depth=10, random_state=42)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    tmp_path = f"/tmp/model_{model_type}.pkl"
    with open(tmp_path, 'wb') as f:
        pickle.dump(model, f)
    return mae

def choose_best_model(**kwargs):
    ti = kwargs['ti']
    mae_rf = ti.xcom_pull(task_ids='train_rf')
    mae_gb = ti.xcom_pull(task_ids='train_gb')
    mae_ridge = ti.xcom_pull(task_ids='train_ridge')
    mae_tree = ti.xcom_pull(task_ids='train_tree')
    
    results = {
        'Random Forest': mae_rf,
        'Gradient Boosting': mae_gb,
        'Ridge Regression': mae_ridge,
        'Decision Tree': mae_tree
    }
    
    best_model_name = min(results, key=results.get)
    best_mae = results[best_model_name]
    
    # Сохраняем модель
    model_slug = best_model_name.lower().replace(" ", "_")
    shutil.copy(f"/tmp/model_{model_slug}.pkl", BEST_MODEL_PATH)
    
    # ОТПРАВЛЯЕМ УВЕДОМЛЕНИЕ В ТЕЛЕГРАМ
    message = (
        f"🚀 Pipeline Finished Successfully!\n\n"
        f"🏆 Winner: {best_model_name}\n"
        f"📉 MAE: ${best_mae:.2f}\n\n"
        f"📊 All Results:\n"
    )
    for name, mae in results.items():
        message += f"- {name}: ${mae:.2f}\n"
    
    send_telegram_message(message)
    print(f"Message sent. Winner: {best_model_name}")

# --- ОПРЕДЕЛЕНИЕ DAG ---
with DAG(
    'nyc_taxi_weather_ml_v3', # Обновил версию
    default_args=default_args,
    description='Complex ML Pipeline with Telegram Alerts',
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False
) as dag:

    create_table = PostgresOperator(
        task_id='create_table',
        postgres_conn_id='postgres_default',
        sql="CREATE TABLE IF NOT EXISTS nyc_taxi_weather (pickup_datetime TIMESTAMP, distance FLOAT, price FLOAT, date_hour TIMESTAMP, temperature FLOAT, precipitation FLOAT, wind_speed FLOAT, is_holiday INT);"
    )

    t1_taxi = PythonOperator(task_id='extract_taxi_data', python_callable=extract_transform_taxi)
    t2_weather = PythonOperator(task_id='extract_weather_api', python_callable=extract_weather_api)
    t3_merge = PythonOperator(task_id='merge_data', python_callable=merge_and_prepare)
    t4_load = PythonOperator(task_id='load_to_postgres', python_callable=load_to_postgres)
    t4_s3 = PythonOperator(task_id='upload_to_s3', python_callable=upload_to_s3, provide_context=True)
    t5_check = BranchPythonOperator(task_id='check_data_quality', python_callable=check_data_quality)
    
    t6_rf = PythonOperator(task_id='train_rf', python_callable=train_model, op_kwargs={'model_type': 'random_forest'})
    t6_gb = PythonOperator(task_id='train_gb', python_callable=train_model, op_kwargs={'model_type': 'gradient_boosting'})
    t6_ridge = PythonOperator(task_id='train_ridge', python_callable=train_model, op_kwargs={'model_type': 'ridge'})
    t6_tree = PythonOperator(task_id='train_tree', python_callable=train_model, op_kwargs={'model_type': 'decision_tree'})

    t7_select_best = PythonOperator(task_id='select_best_model', python_callable=choose_best_model, trigger_rule='none_failed')
    t_stop = PythonOperator(task_id='stop_pipeline', python_callable=lambda: print("Stopped"))

    create_table >> t1_taxi 
    t1_taxi >> t2_weather >> t3_merge 
    t3_merge >> [t4_load, t4_s3] >> t5_check
    t5_check >> t_stop
    t5_check >> [t6_rf, t6_gb, t6_ridge, t6_tree] >> t7_select_best