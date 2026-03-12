import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. Load training data
# -----------------------------
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()

# -----------------------------
# 2. Load testing data
# -----------------------------
X_test = pd.read_csv("X_test.csv")

# -----------------------------
# 3. Initialize Random Forest
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# 4. Train the model
# -----------------------------
rf_model.fit(X_train, y_train)

# -----------------------------
# 5. Generate predictions
# -----------------------------
predictions = rf_model.predict(X_test)

# -----------------------------
# 6. Save predictions
# -----------------------------
output = pd.DataFrame({
    "predicted_meal_category": predictions
})

output.to_csv("../Evaluation/random_forest_predictions.csv", index=False)

print("Random Forest model trained successfully and predictions are saved")
