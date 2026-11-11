
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "crop_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "prediction_history.csv")

st.set_page_config(
    page_title="AI Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. LOAD MODEL AND DATASET
# ==========================================

@st.cache_resource
def load_model():

    saved_data = joblib.load(MODEL_PATH)

    return saved_data


@st.cache_data
def load_dataset():

    return pd.read_csv(DATA_PATH)


# ==========================================
# 3. SIDEBAR
# ==========================================

st.sidebar.title("🌱 AI Agriculture")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🌾 Crop Prediction",
        "📊 Model Insights",
        "📜 Prediction History"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "AI-powered crop recommendations "
    "using soil and weather parameters."
)


# ==========================================
# 4. LOAD PROJECT FILES
# ==========================================

try:

    if not os.path.exists(MODEL_PATH):
        st.error("Trained model not found. Run train_model.py first.")
        st.stop()

    if not os.path.exists(DATA_PATH):
        st.error("Crop_recommendation.csv not found.")
        st.stop()

    saved_data = load_model()

    model = saved_data["model"]

    features = saved_data.get("features")

    df = load_dataset()

    if features is None:
        features = list(df.columns[:-1])

    if not all(feature in df.columns for feature in features):
        st.error("Model features do not match dataset columns.")
        st.stop()

    if len(model.classes_) < 3:
        st.error("The model must contain at least three crop classes.")
        st.stop()

except Exception as e:

    st.error(f"Error loading project files: {e}")
    st.stop()


# ==========================================
# 5. DASHBOARD HOME
# ==========================================

if page == "🏠 Dashboard":

    st.title("🌱 AI Crop Recommendation System")

    st.subheader(
        "Smart Agriculture | Machine Learning | Crop Prediction"
    )

    st.markdown(
        """
        Welcome to the AI Crop Recommendation Dashboard.

        This application uses a trained Random Forest model
        to recommend suitable crop classes based on soil
        nutrients and weather conditions.
        """
    )

    st.markdown("---")

    # Dataset metrics

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Dataset Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Crop Categories",
        df["label"].nunique() if "label" in df.columns else len(model.classes_)
    )

    col3.metric(
        "Model",
        saved_data.get("model_name", "Trained ML Model")
    )

    st.markdown("---")

    st.subheader("🚀 Dashboard Features")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            ### 🌾 Crop Prediction

            - Enter soil NPK values.
            - Enter temperature and humidity.
            - Enter soil pH and rainfall.
            - Get top 3 crop recommendations.
            - View model scores.
            """
        )

    with col2:

        st.markdown(
            """
            ### 📊 Model Insights

            - Feature importance.
            - Crop distribution.
            - Dataset information.
            - Prediction history.
            - Download history as CSV.
            """
        )


# ==========================================
# 6. CROP PREDICTION
# ==========================================

elif page == "🌾 Crop Prediction":

    st.title("🌾 AI Crop Prediction")

    st.write(
        "Enter your farm's soil and weather information."
    )

    with st.form("crop_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("🧪 Soil Nutrients")

            N = st.number_input(
                "Nitrogen (N)",
                min_value=0.0,
                max_value=150.0,
                value=50.0
            )

            P = st.number_input(
                "Phosphorus (P)",
                min_value=0.0,
                max_value=150.0,
                value=50.0
            )

            K = st.number_input(
                "Potassium (K)",
                min_value=0.0,
                max_value=210.0,
                value=50.0
            )

            ph = st.number_input(
                "Soil pH",
                min_value=0.0,
                max_value=14.0,
                value=6.5,
                step=0.1
            )

        with col2:

            st.subheader("🌦️ Weather Conditions")

            temperature = st.number_input(
                "Temperature (°C)",
                min_value=0.0,
                max_value=50.0,
                value=25.0
            )

            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0
            )

            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                max_value=500.0,
                value=100.0
            )

        submitted = st.form_submit_button(
            "🌱 Predict Top 3 Crops",
            use_container_width=True
        )

    if submitted:

        # Prepare input data

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

            st.error(
                "Unexpected model features. Check crop_model.pkl."
            )

        else:

            input_data = pd.DataFrame(
                [[input_values[feature] for feature in features]],
                columns=features
            )

            if not np.isfinite(input_data.to_numpy()).all():

                st.error("Please enter valid finite numbers.")

            else:

                # Training dataset range validation

                warnings = []

                for feature in features:

                    minimum = df[feature].min()
                    maximum = df[feature].max()

                    value = input_data[feature].iloc[0]

                    if value < minimum or value > maximum:

                        warnings.append(
                            f"{feature}: {value} is outside "
                            f"the training dataset range "
                            f"({minimum:.2f} - {maximum:.2f})"
                        )

                st.session_state["last_prediction"] = None

                if warnings:

                    st.warning(
                        "Unusual input values detected. "
                        "Predictions may be less reliable."
                    )

                    for warning in warnings:
                        st.write("⚠️", warning)

                else:

                    st.success(
                        "All inputs are within training dataset ranges."
                    )

                # Make prediction

                try:

                    probabilities = model.predict_proba(
                        input_data
                    )[0]

                    crop_names = model.classes_

                    top_indices = np.argsort(
                        probabilities
                    )[::-1][:3]

                    top_crops = crop_names[top_indices]

                    top_scores = probabilities[top_indices] * 100

                    # Display results

                    st.markdown("---")

                    st.subheader("🏆 Top 3 Recommended Crops")

                    cols = st.columns(3)

                    for rank, index in enumerate(top_indices):

                        crop = crop_names[index]

                        score = probabilities[index] * 100

                        with cols[rank]:

                            st.metric(
                                f"Rank {rank + 1}",
                                crop.upper(),
                                f"{score:.2f}%"
                            )

                            st.progress(
                                float(probabilities[index])
                            )

                    # Save prediction history

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

                    try:

                        history_df.to_csv(
                            HISTORY_PATH,
                            mode="a",
                            header=not os.path.exists(HISTORY_PATH),
                            index=False
                        )

                        st.success(
                            "Prediction saved to history successfully!"
                        )

                    except OSError as e:

                        st.warning(
                            f"Could not save prediction history: {e}"
                        )

                    st.session_state["last_prediction"] = {
                        "input": input_values,
                        "crops": top_crops.tolist(),
                        "scores": top_scores.tolist()
                    }

                    # Prediction chart

                    chart_df = pd.DataFrame({

                        "Crop": top_crops,

                        "Model Score (%)": top_scores

                    })

                    fig = px.bar(
                        chart_df,
                        x="Crop",
                        y="Model Score (%)",
                        title="Top 3 Crop Recommendation Scores",
                        text="Model Score (%)"
                    )

                    fig.update_traces(
                        texttemplate="%{text:.2f}%",
                        textposition="outside"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.info(
                        "These are model-estimated class probabilities, "
                        "not guaranteed crop-success probabilities."
                    )

                except Exception as e:

                    st.error(f"Prediction failed: {e}")


# ==========================================
# 7. MODEL INSIGHTS
# ==========================================

elif page == "📊 Model Insights":

    st.title("📊 Model Insights")

    st.subheader("Feature Importance")

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({

            "Feature": features,

            "Importance": model.feature_importances_ * 100

        }).sort_values(
            by="Importance",
            ascending=False
        )

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Overall Random Forest Feature Importance"
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Feature importance describes overall model behaviour, "
            "not the explanation for an individual prediction."
        )

    else:

        st.info(
            "Feature importance is not available for this model."
        )

    st.markdown("---")

    # Crop distribution

    if "label" in df.columns:

        st.subheader("🌾 Crop Distribution")

        crop_counts = (
            df["label"]
            .value_counts()
            .reset_index()
        )

        crop_counts.columns = [
            "Crop",
            "Records"
        ]

        fig = px.bar(
            crop_counts,
            x="Crop",
            y="Records",
            title="Number of Records Per Crop"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ==========================================
# 8. PREDICTION HISTORY
# ==========================================

elif page == "📜 Prediction History":

    st.title("📜 Prediction History")

    st.write(
        "View previous crop recommendations saved by the application."
    )

    if os.path.exists(HISTORY_PATH):

        try:

            history = pd.read_csv(HISTORY_PATH)

            if history.empty:

                st.info("No prediction history available yet.")

            else:

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Predictions",
                    len(history)
                )

                col2.metric(
                    "Unique Top Crops",
                    history["Top_1_Crop"].nunique()
                )

                col3.metric(
                    "Most Recommended Crop",
                    history["Top_1_Crop"].mode().iloc[0]
                )

                st.markdown("---")

                st.subheader("📋 Prediction Records")

                st.dataframe(
                    history,
                    use_container_width=True,
                    hide_index=True
                )

                # Download CSV

                csv_data = history.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Prediction History",
                    data=csv_data,
                    file_name="prediction_history.csv",
                    mime="text/csv"
                )

                # History chart

                if "Top_1_Crop" in history.columns:

                    st.subheader(
                        "Most Frequently Recommended Crops"
                    )

                    crop_counts = (
                        history["Top_1_Crop"]
                        .value_counts()
                        .reset_index()
                    )

                    crop_counts.columns = [
                        "Crop",
                        "Count"
                    ]

                    fig = px.bar(
                        crop_counts,
                        x="Crop",
                        y="Count",
                        title="Top Recommendation Frequency"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

        except Exception as e:

            st.error(
                f"Unable to read prediction history: {e}"
            )

    else:

        st.info(
            "No prediction history found. "
            "Make your first prediction to create history."
        )


# ==========================================
# 9. FOOTER
# ==========================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI Crop Recommendation System | Python + Scikit-learn + Streamlit"
)
