# NYC Taxi Price Prediction Pipeline 🚖

An end-to-end Data Engineering pipeline that ingests NYC taxi trip data, enriches it with weather data, and trains a Machine Learning model to predict trip fares.

The project is containerized using **Docker**, orchestrated by **Apache Airflow**, and mimics a real-world cloud environment using **AWS S3** and **PostgreSQL**.

---

## 🏗 Architecture

The pipeline follows an ELT (Extract, Load, Transform) pattern:

1.  **Ingestion:** Raw taxi data (CSV) and weather data are extracted.
2.  **Staging:** Data is loaded into a local **PostgreSQL** database.
3.  **Transformation:** SQL and Pandas are used to clean, merge, and feature engineer the dataset.
4.  **Machine Learning:** A Random Forest Regressor is trained on the processed data.
5.  **Storage:** The processed dataset and model metrics are uploaded to an **AWS S3** bucket for downstream usage.

## 🛠 Tech Stack

* **Language:** Python 3.8+
* **Orchestration:** Apache Airflow 2.x
* **Containerization:** Docker & Docker Compose
* **Database:** PostgreSQL 13
* **Cloud Storage:** AWS S3
* **Libraries:** Pandas, Scikit-Learn, SQLAlchemy, Psycopg2

---
## 🌟 New Features (v2.0)

### 🤖 AutoML & Parallel Training
The pipeline now trains **4 different Machine Learning models** in parallel to find the best performer:
1.  **Random Forest** (Robust baseline)
2.  **Gradient Boosting** (High accuracy)
3.  **Ridge Regression** (Linear baseline)
4.  **Decision Tree** (Interpretability)

The system automatically compares their **MAE (Mean Absolute Error)** and promotes the best model to production.

### 📱 Real-time Alerts
Integrated with **Telegram Bot API** for instant notifications.
- Sends a report immediately after pipeline completion.
- Displays the winning model and performance metrics directly in your chat.
- Zero-downtime monitoring.

### 🛡️ Security
- All credentials are managed via **Environment Variables**.
- Project is secured against accidental secret exposure using `.env` and `.gitignore`.

---

## 📂 Project Structure

```text
nyc_taxi_project/
├── dags/
│   ├── data/                  # Place your raw taxi_data.csv here
│   └── nyc_taxi_price_predictor.py  # Main DAG logic
├── logs/                      # Airflow logs
├── plugins/                   # Airflow plugins
├── .env                       # Environment variables (not in repo)
├── docker-compose.yaml        # Container orchestration
├── Dockerfile                 # Custom Airflow image
├── requirements.txt           # Python dependencies
└── README.md
```

## Installation & Starting
1. Prerequisites
Docker & Docker Compose installed.

AWS Account (Access Key & Secret Key) with S3 write permissions.

Git.

#2. Clone the Repository
Bash

git clone [https://github.com/Reigenium/nyc_taxi_project.git](https://github.com/Reigenium/nyc_taxi_project.git)
cd nyc_taxi_project
3. Setup Environment Variables
Create a .env file in the root directory to set the Airflow user permissions:

Bash

echo "AIRFLOW_UID=50000" > .env
#4. Add the Dataset
Important: The raw data file (taxi_data.csv) is too large for GitHub and is not included in the repo.

Download your NYC Taxi CSV file.

Place it inside the dags/data/ folder.

If running on a remote server (EC2), upload it via SCP:

Bash

scp -i "your-key.pem" "path/to/taxi_data.csv" ubuntu@YOUR_SERVER_IP:~/nyc_taxi_project/dags/data/
#5. Start the Project
Run the following command to build the image and start the containers:

Bash

docker-compose up -d --build
Wait 1-2 minutes for the Webserver and Scheduler to initialize.

##⚙️Configuration (Required)
Once Docker is running, access the Airflow UI at http://localhost:8080 (or http://YOUR_EC2_IP:8080). Login: airflow / airflow

You must manually configure two connections in Admin -> Connections for the pipeline to run.

#1. PostgreSQL Connection
This connects Airflow to the local staging database.

Conn Id: postgres_default

Conn Type: Postgres

Host: postgres (Note: use the service name, not localhost)

Login: airflow

Password: airflow

Port: 5432

#2. AWS S3 Connection
This connects Airflow to your cloud storage.

Conn Id: aws_s3_conn

Conn Type: Amazon Web Services

Login: <Your AWS Access Key ID>

Password: <Your AWS Secret Access Key>

⚠️AWS Learner Lab Users: If you are using temporary credentials (AWS Academy/Vocareum), you must also add your Session Token in the Extra field as JSON:

#JSON

{
   "aws_session_token": "YOUR_VERY_LONG_SESSION_TOKEN_HERE"
}
(Note: If your AWS session expires, you must update these keys in Airflow to run the pipeline again).

##Usage
In the Airflow UI, find the DAG named nyc_taxi_weather_prediction.

Toggle the switch to ON (Blue).

Click the Play Button (Trigger DAG) under Actions.

Switch to the Graph View to watch the tasks execute in real-time.

Expected Output
S3 Bucket: A new file processed_data_YYYY-MM-DD.csv will appear.

Airflow Logs: Check the train_model task logs to see the model's MAE (Mean Absolute Error).
