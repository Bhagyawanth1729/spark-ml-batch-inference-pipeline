# 🚀 Batch Inference Pipeline with Apache Spark ML

This project demonstrates how to build and run a **batch scoring pipeline** using **Apache Spark MLlib**.  
We train a classification model using Spark ML, save it as a `PipelineModel`, and then use it to run batch inference on a large CSV dataset. The predictions are written to **Parquet format partitioned by date**.

---

## 📌 Project Features

- Load large CSV input data
- Train a Spark ML pipeline (with feature encoding)
- Save and reload a pretrained `PipelineModel`
- Run batch inference at scale
- Write predictions in Parquet format partitioned by `event_date`
- Works locally or on Google Colab

---

## 📂 Project Structure
#
project/
│
├── telco_dataset.csv # Input dataset (generated)
├── generate_telco_dataset.py # Script to generate sample data
├── train_model.py # Train and save Spark ML model
├── batch_inference.py # Batch scoring job
│
├── model/
│ └── pipeline_model/ # Saved Spark ML PipelineModel
│
├── output_predictions/ # Parquet output (partitioned by date)
│
└── README.md
#

---

## 🧾 Dataset

The dataset is Telco-style customer data with features such as:

- `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- `tenure`
- `PhoneService`, `MultipleLines`
- `InternetService`, `OnlineSecurity`, `OnlineBackup`
- `event_date` → used for partitioning output

Example:
```csv
customer_id,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,event_date
C0000001,Male,0,Yes,No,12,Yes,No,DSL,Yes,No,2025-12-11

⚙️ Requirements
Java 8+
Python 3.8+
Apache Spark 3.x
PySpark
Install PySpark:


pip install pyspark
▶️ Running on Google Colab
Install Spark & PySpark in Colab:


!apt-get install openjdk-8-jdk-headless -qq
!pip install pyspark
Upload telco_dataset.csv and notebooks/scripts.

Create Spark session inside Colab:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("BatchInference").getOrCreate()
🧪 Step 1: Generate Dataset (Optional)
bash
Copy code
python generate_telco_dataset.py --output telco_dataset.csv --rows 500000
This creates a large CSV with unique customer IDs and random features.

🏗️ Step 2: Train & Save Model
Train a Spark ML pipeline and save it:
spark-submit train_model.py
This will:
Encode categorical features
Assemble features
Train a Logistic Regression classifier

Save the model to:
model/pipeline_model/
🔮 Step 3: Run Batch Inference
Use the saved model to score the full dataset:


spark-submit batch_inference.py \
  --model_path model/pipeline_model \
  --input_path telco_dataset.csv \
  --output_path output_predictions
📦 Output
Predictions are written in Parquet format partitioned by event_date:

Copy code
output_predictions/
 ├── event_date=2025-12-10/
 ├── event_date=2025-12-11/
 ├── event_date=2025-12-12/
 └── ...

Each file contains:
customer_id
prediction
probability
event_date

Example:
+-----------+----------+--------------------+----------+
|customer_id|prediction|probability         |event_date|
+-----------+----------+--------------------+----------+
|C0000004   |1.0       |[0.49, 0.51]         |2025-12-11|

🔍 Verify Output
spark.read.parquet("output_predictions").show(10)
+-----------+----------+--------------------+----------+
|customer_id|prediction|         probability|event_date|
+-----------+----------+--------------------+----------+
|   C0000004|       1.0|[0.49637088771665...|2025-12-11|
|   C0000015|       1.0|[0.49684819803810...|2025-12-11|
|   C0000028|       1.0|[0.49887567753226...|2025-12-11|
|   C0000039|       0.0|[0.50271165365204...|2025-12-11|
|   C0000065|       1.0|[0.49574045654020...|2025-12-11|
+-----------+----------+--------------------+----------+
only showing top 5 rows
