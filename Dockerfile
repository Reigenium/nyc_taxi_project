FROM apache/airflow:2.8.1
USER root
# Устанавливаем компиляторы (нужны для некоторых python-библиотек)
RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow
# Устанавливаем библиотеки: 
# pandas - работа с данными
# scikit-learn - ML
# requests - запросы к API
# psycopg2-binary - драйвер Postgres
# pyarrow - ускоряет чтение CSV/Parquet
# holidays - список праздников США 
RUN pip install --no-cache-dir pandas scikit-learn requests psycopg2-binary pyarrow holidays