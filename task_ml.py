from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import time

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Local Hadoop Processing Debug") \
    .getOrCreate()

# Path to your local Hadoop HDFS data
local_hdfs_path = "hdfs://localhost:9000/user/adv/csv_data/"

# Load, process, and train the model
def process_and_train_local(hdfs_path):
    print(f"Processing data from: {hdfs_path}")
    
    # Load the CSV data from HDFS
    start_time = time.time()
    data = spark.read.csv(hdfs_path, header=True, inferSchema=True)
    load_time = time.time() - start_time
    print(f"Data Load Time: {load_time:.2f} seconds")
    
    # Check total rows
    total_rows = data.count()
    print(f"Total Rows in Dataset: {total_rows}")

    # Debug: Display schema
    data.printSchema()

    # Debug: Check missing values column-wise
    for col in data.columns:
        missing_count = data.filter(data[col].isNull() | (data[col] == "")).count()
        print(f"Column '{col}' has {missing_count} missing values")

    # Drop rows with missing values in critical columns
    critical_columns = ["runs_total", "over", "inning"]  # Replacing 'ball' with 'inning'
    print(f"Filtering rows based on critical columns: {critical_columns}")
    data = data.dropna(subset=critical_columns)

    rows_after_drop = data.count()
    print(f"Rows After Dropping NA: {rows_after_drop}")

    if rows_after_drop == 0:
        print("No rows left after dropping missing values in critical columns.")
        return load_time, 0

    # Select numeric columns for features
    numeric_columns = [col for col, dtype in data.dtypes if dtype in ('int', 'double')]
    label_column = "runs_total"  # Replace with your target column for prediction

    if label_column not in numeric_columns:
        raise ValueError(f"Label column '{label_column}' must be numeric.")

    numeric_columns.remove(label_column)

    print(f"Numeric Columns for Features: {numeric_columns}")

    # Assemble features into a single vector column
    assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features")
    data = assembler.transform(data)

    # Split the data into training and testing sets
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

    train_rows = train_data.count()
    test_rows = test_data.count()
    print(f"Training Rows: {train_rows}, Testing Rows: {test_rows}")

    if train_rows == 0:
        raise ValueError("Training dataset is empty.")

# Train the model with regularization (L2 regularization)
    lr = LinearRegression(featuresCol="features", labelCol=label_column, regParam=0.1, maxIter=100)

    start_time = time.time()
    lr_model = lr.fit(train_data)
    training_time = time.time() - start_time

    print(f"Training Time: {training_time:.2f} seconds")

    # Evaluate the model on test data
    test_results = lr_model.evaluate(test_data)
    print(f"Root Mean Squared Error (RMSE): {test_results.rootMeanSquaredError}")
    # print(f"R2 Score: {test_results.r2-0.10}")

    return load_time, training_time

# Process data on local Hadoop
try:
    print("== Local Hadoop ==")
    local_load_time, local_training_time = process_and_train_local(local_hdfs_path)

    # Print results
    print("\n== Local Hadoop Performance ==")
    print(f"Data Load Time: {local_load_time:.2f} seconds")
    print(f"Training Time: {local_training_time:.2f} seconds")
except Exception as e:
    print(f"Error: {e}")

# Stop the Spark session
spark.stop()
