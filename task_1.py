import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time

# Measure the time taken for the entire process
start_time = time.time()

# Folder containing all CSV files
folder_path = '/Users/pateldhrit/Desktop/Hadoop-Project/output'  # Replace with your folder path

# Step 1: Load and concatenate all CSV files
all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
data_list = [pd.read_csv(file) for file in all_files]
full_data = pd.concat(data_list, ignore_index=True)

# Step 2: Preprocess data
data_cleaned = full_data.drop(columns=['match_id', 'date', 'ball', 'wicket_player', 'extras_type', 'wicket_type'])

# One-hot encode categorical columns
categorical_columns = ['team1', 'team2', 'inning', 'batsman', 'bowler', 'non_striker', 
                       'match_type', 'gender', 'batting_team', 'bowling_team']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(data_cleaned[categorical_columns])

# Combine encoded features with numerical columns
numerical_columns = ['over', 'runs_batsman', 'runs_extras']
X = np.hstack((encoded_features, data_cleaned[numerical_columns].values))
y = data_cleaned['runs_total']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Step 4: Visualizations
# Plot 1: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Runs Total')
plt.ylabel('Predicted Runs Total')
plt.title('Actual vs Predicted Runs Total')
plt.grid(True)
plt.show()

# Plot 2: Feature Importance
feature_importances = model.feature_importances_
importance_indices = np.argsort(feature_importances)[-10:]  # Top 10 features
top_features = np.array(encoder.get_feature_names_out(categorical_columns).tolist() + numerical_columns)[importance_indices]

plt.figure(figsize=(12, 6))
plt.barh(range(len(top_features)), feature_importances[importance_indices], color='green')
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances')
plt.grid(True)
plt.show()

# Measure the total time taken
end_time = time.time()
print(f"Total Time Taken: {end_time - start_time:.2f} seconds")
