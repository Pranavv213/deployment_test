import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# Generate random data
np.random.seed(42)
n_samples = 1000
n_features = 4

X = np.random.rand(n_samples, n_features) * 100
true_coeffs = np.random.randn(n_features) * 1000
intercept = np.random.randint(1000000, 5000000)
y = X @ true_coeffs + intercept + np.random.randn(n_samples) * 1000000
y = y + 45000000
y = np.abs(y)

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Save the model to a pickle file
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'linear_regression_model.pkl'")

# Sample input (5 samples, 4 features each)
sample_input = np.array([
    [45.67, 23.45, 78.90, 12.34],
    [67.89, 34.56, 12.34, 90.12],
    [12.34, 89.01, 56.78, 45.67],
    [78.90, 56.78, 34.56, 23.45],
    [34.56, 90.12, 45.67, 78.90]
])

# Make predictions
predictions = model.predict(sample_input)

print("\nSample Input (5 samples, 4 features each):")
print(sample_input)
print("\nPredictions (target values):")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: {pred:.2f}")