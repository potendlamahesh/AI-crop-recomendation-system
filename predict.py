
import os
import joblib
import numpy as np
import pandas as pd

from datetime import datetime


# ==========================================
# 1. PROJECT FILE PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "crop_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "prediction_history.csv")


# ==========================================
# 2. MAIN PROGRAM
# ==========================================

print("\n====================================")
print("       AI CROP RECOMMENDATION")
print("====================================")


try:

    # ======================================
    # 3. LOAD TRAINED MODEL
    # ======================================

    saved_data = joblib.load(MODEL_PATH)

    model = saved_data["model"]

    features = saved_data.get("features")

    # Load original dataset
    df = pd.read_csv(DATA_PATH)

    if features is None:
        features = list(df.columns[:-1])

    if not all(feature in df.columns for feature in features):
        raise ValueError(
            "Model features do not match the dataset columns."
        )

    if len(model.classes_) < 3:
        raise ValueError(
            "The model must contain at least three crop classes."
        )

    # ======================================
    # 4. GET FARMER INPUTS
    # ======================================

    print("\nEnter Soil and Weather Information:\n")

    N = float(input("Enter Nitrogen (N): "))
    P = float(input("Enter Phosphorus (P): "))
    K = float(input("Enter Potassium (K): "))

    temperature = float(input("Enter Temperature (C): "))
    humidity = float(input("Enter Humidity (%): "))

    ph = float(input("Enter Soil pH: "))
    rainfall = float(input("Enter Rainfall (mm): "))

    # Map inputs to their feature names
    input_values = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

    if set(features) != set(input_values):
        raise ValueError(
            "Unexpected model features. "
            "Check the feature names in crop_model.pkl."
        )

    input_data = pd.DataFrame(
        [[input_values[feature] for feature in features]],
        columns=features
    )

    # ======================================
    # 5. BASIC INPUT VALIDATION
    # ======================================

    if not np.isfinite(input_data.to_numpy()).all():
        raise ValueError(
            "Inputs must be finite numbers."
        )

    if not (0 <= N <= 150 and
            0 <= P <= 150 and
            0 <= K <= 210):

        raise ValueError(
            "N, P or K value is outside the accepted range."
        )

    if not (0 <= temperature <= 50):
        raise ValueError(
            "Temperature must be between 0 and 50 C."
        )

    if not (0 <= humidity <= 100):
        raise ValueError(
            "Humidity must be between 0 and 100."
        )

    if not (0 <= ph <= 14):
        raise ValueError(
            "Soil pH must be between 0 and 14."
        )

    if not (0 <= rainfall <= 500):
        raise ValueError(
            "Rainfall must be between 0 and 500 mm."
        )

    print("\nInput validation successful!")

    # ======================================
    # 6. TRAINING DATA RANGE VALIDATION
    # ======================================

    warnings = []

    print("\n====================================")
    print("       SUITABILITY VALIDATION")
    print("====================================")

    for feature in features:

        minimum = df[feature].min()
        maximum = df[feature].max()

        value = input_data[feature].iloc[0]

        if value < minimum or value > maximum:

            warning = (
                f"{feature}: {value} is outside the "
                f"training dataset range "
                f"({minimum:.2f} - {maximum:.2f})"
            )

            warnings.append(warning)

    if warnings:

        print("\nWARNING: Unusual input values detected!")

        for warning in warnings:
            print("-", warning)

        print("\nPredictions may be less reliable.")

    else:

        print(
            "\nAll inputs are within the training dataset ranges."
        )

    # ======================================
    # 7. PREDICT CROPS
    # ======================================

    probabilities = model.predict_proba(input_data)[0]

    crop_names = model.classes_

    # Find top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]

    top_crops = crop_names[top_indices]

    top_scores = probabilities[top_indices] * 100

    # ======================================
    # 8. DISPLAY TOP 3 RECOMMENDATIONS
    # ======================================

    print("\n====================================")
    print("       TOP 3 RECOMMENDED CROPS")
    print("====================================")

    for rank, index in enumerate(top_indices, start=1):

        crop = crop_names[index]

        confidence = probabilities[index] * 100

        print(
            f"{rank}. {crop.upper()} - {confidence:.2f}%"
        )

    # ======================================
    # 9. MODEL FEATURE IMPORTANCE
    # ======================================

    print("\n====================================")
    print("       MODEL FEATURE IMPORTANCE")
    print("====================================")

    if hasattr(model, "feature_importances_"):

        feature_importance = pd.DataFrame({

            "Feature": features,

            "Importance": model.feature_importances_

        }).sort_values(
            by="Importance",
            ascending=False
        )

        for rank, row in enumerate(
            feature_importance.itertuples(index=False),
            start=1
        ):

            percentage = row.Importance * 100

            print(
                f"{rank}. {row.Feature.capitalize():<15} "
                f"- {percentage:.2f}%"
            )

    else:

        print(
            "Feature importance is not available for this model."
        )

    # ======================================
    # 10. SAVE PREDICTION HISTORY
    # ======================================

    history_data = {

        "Date_Time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "Nitrogen": N,
        "Phosphorus": P,
        "Potassium": K,

        "Temperature": temperature,
        "Humidity": humidity,

        "Soil_pH": ph,
        "Rainfall": rainfall,

        "Top_1_Crop": top_crops[0],
        "Top_1_Score": round(top_scores[0], 2),

        "Top_2_Crop": top_crops[1],
        "Top_2_Score": round(top_scores[1], 2),

        "Top_3_Crop": top_crops[2],
        "Top_3_Score": round(top_scores[2], 2)

    }

    history_df = pd.DataFrame([history_data])

    # Append new prediction without deleting old records
    try:

        history_df.to_csv(

            HISTORY_PATH,

            mode="a",

            header=not os.path.exists(HISTORY_PATH),

            index=False

        )

        print("\n====================================")
        print("       PREDICTION HISTORY")
        print("====================================")

        print("\nPrediction saved successfully!")

        print(
            "\nHistory file: prediction_history.csv"
        )

    except OSError as e:

        print(
            "\nWarning: Could not save prediction history:",
            e
        )

    # ======================================
    # 11. FINAL MESSAGE
    # ======================================

    print("\n====================================")
    print("       PREDICTION COMPLETED")
    print("====================================")

    print("\nThank you for using AI Crop Recommendation!")

    print("\nNote:")
    print(
        "Model scores are estimates, not guarantees "
        "of successful crop growth."
    )

    print(
        "Check local soil conditions, season and "
        "agricultural guidance before planting."
    )

    print("\n====================================")


# ==========================================
# 12. ERROR HANDLING
# ==========================================

except ValueError as e:

    print("\nInvalid input:", e)

except FileNotFoundError as e:

    print("\nMissing required file:", e)

except KeyError as e:

    print("\nModel loading error:", e)

except Exception as e:

    print("\nUnexpected error:", e)