# AI Crop Recommendation System

A web app that suggests the top 3 crops for a farm, based on soil and weather values. It uses a Random Forest model trained on a public crop dataset and runs as a Streamlit app.

**Live app:** https://ai-crop-recomendation-system-dhxswnwf9hsmpxlnuns9h7.streamlit.app/

## Project overview

You enter seven values (soil nutrients, soil pH, temperature, humidity and rainfall). The app returns the three crops the model scores highest, with a short check of how your inputs compare with the training data.

## Problem statement

Choosing a crop is hard when you have to weigh many soil and weather factors at once. This project shows how machine learning can give a quick, data-based starting point.

It is a learning and demonstration project. It is **not** a replacement for local agricultural advice.

## Objectives

- Train a model that recommends a crop from seven inputs.
- Show the top 3 crops with the model's scores.
- Warn the user when inputs look unusual.
- Explain the model with simple charts.
- Make it free to use for everyone, on phones and desktops.

## Technologies used

Python, Streamlit, Scikit-learn, Pandas, NumPy, Plotly, Joblib, Git and GitHub, Streamlit Community Cloud.

## Dataset

`Crop_recommendation.csv`

| Item | Value |
|---|---|
| Records | 2,200 |
| Columns | 8 (7 inputs + crop label) |
| Crops | 22 |
| Records per crop | 100 |

Inputs: Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, soil pH, rainfall.

## Machine learning algorithm

A **Random Forest classifier** (Scikit-learn). It was chosen after comparing it with a Decision Tree.

## Model evaluation

| Model | Test-split accuracy |
|---|---|
| Decision Tree | 97.95% |
| Random Forest (used) | 99.55% |

**Important:** this is accuracy on a held-out part of the same dataset. It is **not** a guarantee that a recommendation will work on a real farm. The dataset is clean and balanced, and real fields are messier.

The app reads the accuracy from the saved model file (`crop_model.pkl`). If the file has no accuracy saved, the app shows "Not available". It never invents a number.

## Application features

- **Dashboard:** project summary, dataset facts, model details.
- **Crop Prediction:** top 3 crops, score gap analysis, input warnings, and a per-crop check against the ranges seen in the training data.
- **Model Insights:** feature importance and crop distribution. Feature importance shows what the model uses. It does not prove cause and effect.
- **Prediction History:** your predictions from the current visit, with a CSV download.
- Works on phones, tablets and desktops.

### About prediction history and privacy

History is stored **per visitor, in the browser session only**. Other visitors cannot see it, and nothing is written to a shared file. When the session ends (for example, the tab is closed or the app restarts), the history is gone. Download the CSV if you want to keep it.

## Installation

You need Python 3.11 or newer (developed on 3.13).

```bash
git clone https://github.com/potendlamahesh/AI-crop-recomendation-system.git
cd AI-crop-recomendation-system
python -m venv venv
```

Activate the environment:

```bash
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run predict.py
```

The app opens at http://localhost:8501.

To retrain the model (this overwrites `crop_model.pkl`, so keep a copy first):

```bash
python train_model.py
```

## Project structure

```text
.
├── predict.py                # Streamlit app (main file)
├── train_model.py            # Trains and saves the model
├── crop_model.pkl            # Saved model + metadata
├── Crop_recommendation.csv   # Training dataset
├── requirements.txt          # Pinned packages
├── .streamlit/config.toml    # Green and white theme
├── .gitignore
└── README.md
```

## Deployment (Streamlit Community Cloud)

1. Push the project to GitHub.
2. Open https://share.streamlit.io and choose the repository.
3. Set the main file to `predict.py`.
4. In **Advanced settings**, pick a Python version (3.13 matches the one this project was tested with).
5. Deploy. Any later push to the branch redeploys the app.

Anyone with the app link can use it. They do not need a GitHub account.

## Screenshots

_Add screenshots here (Dashboard, Crop Prediction, Model Insights, Prediction History)._

## Limitations

- The model learns from one dataset. It does not know your local soil, season, water supply, market prices or pests.
- The 99.55% score is a test-split score, not real-world accuracy.
- The scores are model outputs. They are not calibrated probabilities.
- Only 22 crops are supported.
- Prediction history is lost when the session ends.

## Future improvements

- Optional saved history with a free database and user accounts.
- Test the model on data from real farms.
- Add calibration for the model scores.
- Add more crops and regional data.
- Add Telugu and Hindi language options.

## Author

**Mahesh Potendla**

- Portfolio: https://maheshp.in
- GitHub: https://github.com/potendlamahesh

## Contact

Please open an issue on the GitHub repository, or use the contact details on the portfolio site.
