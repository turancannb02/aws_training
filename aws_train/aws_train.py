import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib

# Read data
data = pd.read_csv('/Users/turanbuyukkamaci/Downloads/powerconsumption.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%m/%d/%Y %H:%M')

# Set 'Datetime' as the index for resampling
data.set_index('Datetime', inplace=True)

# Extract time-based features
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month

# Creating a lagged feature
data['PowerConsumption_Zone1_lag1'] = data['PowerConsumption_Zone1'].shift(1)

# Creating a rolling mean feature
window_size = 24  # change to the number of periods you want to consider
data['PowerConsumption_Zone1_rolling_mean'] = data['PowerConsumption_Zone1'].rolling(window_size).mean()

# Drop missing values
data = data.dropna()

# Plot 'PowerConsumption_Zone1' over time
plt.figure(figsize=(10, 6))
plt.plot(data['PowerConsumption_Zone1'])
plt.title('Power Consumption Zone1 Over Time')
plt.xlabel('Time')
plt.ylabel('Power Consumption Zone1')
plt.show(block=True)

# Separate data into features and target variable
X = data.drop(columns='PowerConsumption_Zone1')
y = data['PowerConsumption_Zone1']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model using LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Save the model
joblib.dump(model, 'model.joblib')

# Evaluate model
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Root Mean Squared Error: {rmse}')

# Perform cross-validation
cross_val_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores: ", cross_val_scores)
