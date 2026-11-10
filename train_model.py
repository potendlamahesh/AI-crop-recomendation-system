
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("Crop_recommendation.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())

# 2. Separate input features and target
X = df.drop("label", axis=1)
y = df["label"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

# 5. Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# 6. Compare models
print("\nMODEL COMPARISON")
print("-------------------------")
print("Decision Tree Accuracy:", round(dt_accuracy * 100, 2), "%")
print("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")

# 7. Select the better model
if rf_accuracy >= dt_accuracy:
    best_model = rf_model
    model_name = "Random Forest"
else:
    best_model = dt_model
    model_name = "Decision Tree"

# 8. Evaluate selected model
best_predictions = best_model.predict(X_test)

print("\nSelected Model:", model_name)
print("\nClassification Report:")
print(classification_report(y_test, best_predictions))

# 9. Save trained model, feature names, and accuracy
best_accuracy = max(dt_accuracy, rf_accuracy)

joblib.dump(
    {
        "model": best_model,
        "features": list(X.columns),
        "model_name": model_name,
        "accuracy": float(best_accuracy),
        "accuracy_type": "test_split_accuracy"
    },
    "crop_model.pkl"
)

print("\nModel saved successfully as crop_model.pkl")
print("Saved Model Accuracy:", round(best_accuracy * 100, 2), "%")
print("\nModel saved successfully as crop_model.pkl")
