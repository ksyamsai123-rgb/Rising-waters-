import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Dataset load
data = pd.read_excel("dataset/flooddataset.csv.xlsx")

# Features and Target
X = data.drop("flood", axis=1)
y = data["flood"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "model/flood_model.pkl")
print("Model saved successfully!")