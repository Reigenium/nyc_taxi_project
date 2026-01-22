from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
# 🔥 НОВОЕ: Импорт для работы с S3
from airflow.providers.amazon.aws.hooks.s3 import S3Hook 

from datetime import datetime, timedelta
import pandas as pd
import requests
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# --- КОНФИГУРАЦИЯ ---
DATA_PATH = '/opt/airflow/dags/data/taxi_data.csv'
PROCESSED_PATH = '/tmp/merged_nyc_data.csv'
MODEL_PATH = '/opt/airflow/dags/data/price_model.pkl'

# 🔥 ВПИШИ СЮДА ИМЯ СВОЕГО БАКЕТА
BUCKET_NAME = 'taxi-price-prediction-data' 

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# --- ФУНКЦИИ ---

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
    df.to_parquet('/tmp/taxi_clean.parquet')
    
    min_date = df['date_hour'].min().strftime('%Y-%m-%d')
    max_date = df['date_hour'].max().strftime('%Y-%m-%d')
    return {'min_date': min_date, 'max_date': max_date}

def extract_weather_api(**kwargs):
    ti = kwargs['ti']
    dates = ti.xcom_pull(task_ids='extract_taxi_data')
    print(f"Запрашиваем погоду с {dates['min_date']} по {dates['max_date']}")
    
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
    print(f"Итоговый датасет: {merged.shape[0]} строк.")

def load_to_postgres(**kwargs):
    df = pd.read_csv(PROCESSED_PATH)
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = pg_hook.get_sqlalchemy_engine()
    df.to_sql('nyc_taxi_weather', engine, if_exists='replace', index=False)
    print("Данные успешно загружены в Postgres.")

# 🔥 НОВОЕ: Функция загрузки в S3
def upload_to_s3(**kwargs):
    """
    Загружает обработанный файл в S3 бакет
    """
    # Формируем уникальное имя файла с датой запуска (чтобы не затирать старые)
    execution_date = kwargs['ds'] 
    s3_key = f"processed_data_{execution_date}.csv"
    
    hook = S3Hook(aws_conn_id='aws_s3_conn') # Используем подключение, которое создали в UI
    
    hook.load_file(
        filename=PROCESSED_PATH,
        key=s3_key,
        bucket_name=BUCKET_NAME,
        replace=True
    )
    print(f"Файл {PROCESSED_PATH} загружен в S3: {BUCKET_NAME}/{s3_key}")

def check_data_quality(**kwargs):
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    records = pg_hook.get_first("SELECT COUNT(*) FROM nyc_taxi_weather")
    count = records[0]
    if count > 1000:
        return 'train_model'
    else:
        return 'stop_pipeline'

def train_ml_model(**kwargs):
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    df = pg_hook.get_pandas_df("SELECT * FROM nyc_taxi_weather")
    X = df[['distance', 'temperature', 'precipitation', 'wind_speed']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"MAE: ${mae:.2f}, R2: {r2:.4f}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    return f"MAE: {mae}"

# --- ОПРЕДЕЛЕНИЕ DAG ---

with DAG(
    'nyc_taxi_weather_prediction',
    default_args=default_args,
    description='Predict Taxi Fares based on Weather',
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False
) as dag:

    create_table = PostgresOperator(
        task_id='create_table',
        postgres_conn_id='postgres_default',
        sql="""
            CREATE TABLE IF NOT EXISTS nyc_taxi_weather (
                pickup_datetime TIMESTAMP,
                distance FLOAT,
                price FLOAT,
                date_hour TIMESTAMP,
                temperature FLOAT,
                precipitation FLOAT,
                wind_speed FLOAT
            );
        """
    )

    t1_taxi = PythonOperator(
        task_id='extract_taxi_data',
        python_callable=extract_transform_taxi
    )
    
    t2_weather = PythonOperator(
        task_id='extract_weather_api',
        python_callable=extract_weather_api
    )

    t3_merge = PythonOperator(
        task_id='merge_data',
        python_callable=merge_and_prepare
    )

    t4_load = PythonOperator(
        task_id='load_to_postgres',
        python_callable=load_to_postgres
    )

    # 🔥 НОВОЕ: Задача загрузки в S3
    t4_s3_upload = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_to_s3,
        provide_context=True # Нужно, чтобы получить дату запуска
    )

    t5_branch = BranchPythonOperator(
        task_id='check_data_quality',
        python_callable=check_data_quality
    )

    t6_train = PythonOperator(
        task_id='train_model',
        python_callable=train_ml_model
    )

    t7_stop = PythonOperator(
        task_id='stop_pipeline',
        python_callable=lambda: print("Недостаточно данных для обучения.")
    )

    # 🔥 ОБНОВЛЕННАЯ ЛОГИКА: Параллельный запуск
    # Сначала создаем таблицу
    create_table >> t1_taxi 
    
    # Готовим данные
    t1_taxi >> t2_weather >> t3_merge 
    
    # После объединения запускаем ДВЕ задачи одновременно: 
    # 1. Грузим в Postgres
    # 2. Грузим в S3
    t3_merge >> [t4_load, t4_s3_upload] 
    
    # Когда ОБЕ загрузки прошли, проверяем качество
    [t4_load, t4_s3_upload] >> t5_branch
    
    # Ветвление
    t5_branch >> [t6_train, t7_stop]