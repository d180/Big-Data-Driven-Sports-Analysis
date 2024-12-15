from pyspark.sql import SparkSession
import time

# Initialize Spark session
spark = SparkSession.builder.appName("CricketDataAnalysis").getOrCreate()

# Correct HDFS path to your data
input_directory = "hdfs://localhost:9000/user/adv/csv_data/*"  # Adjust the HDFS path accordingly

# Load the CSV files into Spark DataFrame
df = spark.read.csv(input_directory, header=True, inferSchema=True)

# Register the DataFrame as a temporary SQL table
df.createOrReplaceTempView("cricket_data")

# Measure the start time for the SQL query execution
start_time = time.time()

# Run SQL queries to analyze the data
# Example: Total Runs by Each Batsman
result = spark.sql("""
    SELECT batsman, SUM(runs_batsman) AS total_runs
    FROM cricket_data
    GROUP BY batsman
    ORDER BY total_runs DESC
    LIMIT 10
""")

# Show the result
result.show()

# Measure the end time
end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print(f"Query Execution Time: {execution_time} seconds")