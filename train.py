import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
dataset = pd.read_csv('crops_dataset.csv')

# Separate features (X) and target (y)
X = dataset.drop('Crop', axis=1)  # X contains all columns except 'Crop'
y = dataset['Crop']  # y contains only the 'Crop' column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a model (example with RandomForestClassifier)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to disk as .pkl
joblib.dump(model, 'crop_prediction_model.pkl')
