# Student Placement Predictor

A machine learning web app that predicts whether a student is likely to be placed, based on CGPA, IQ, and a profile score. The model is an SVM classifier served through a Flask web interface that also keeps a short history of recent predictions.

## Project Structure

```
.
├── app.py                               # Flask application
├── templates/
│   └── index.html                       # Web interface (form, result, history, chart)
├── model.pkl                            # Trained SVC model
├── students_placement.csv               # Dataset (300 students)
├── student-placement-predictor.ipynb    # Model training and comparison
├── prediction_history.csv               # Log of predictions made in the app
├── requirements.txt                     # Python dependencies
└── README.md
```

## Dataset

`students_placement.csv` contains 300 student records (148 label 0, 152 label 1, so classes are roughly balanced).

| Column | Description | Range in data |
|---|---|---|
| `cgpa` | Cumulative grade point average | 4.6 – 9.3 |
| `iq` | IQ score | 83 – 121 |
| `profile_score` | Profile score | 23 – 100 |
| `placed` | **Target** (1 / 0) | 0 or 1 |

## Methodology

1. **Features / target**: `cgpa`, `iq`, `profile_score` → `placed`.
2. **Split**: 80% train / 20% test (`random_state=2`), giving 240 training and 60 test rows.
3. **Models compared** on the test set:

| Model | Test accuracy |
|---|---|
| Logistic Regression (with StandardScaler) | 88.3% |
| Random Forest | 91.7% |
| **SVC (RBF kernel)** | **93.3%** |

4. **Final model**: `SVC(kernel='rbf')` with default parameters, trained on the unscaled features and saved to `model.pkl`.

## Web App

Built with Flask. The home page shows the input form and the five most recent predictions.

- `GET /` renders the form and recent history
- `POST /predict` reads `cgpa`, `iq`, and `profile_score`, validates them, runs the model, logs the result, and re-renders the page

**Input validation**

- CGPA must be between 0 and 10
- IQ must be greater than 0
- Profile Score must be between 0 and 100

Each prediction is appended to `prediction_history.csv` (`CGPA, IQ, Profile Score, Prediction`).

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
pip install -r requirements.txt
```

Dependencies: `flask`, `numpy`, `pandas`, `scikit-learn`.

> Use the scikit-learn version the model was trained with to avoid pickle compatibility warnings.

### Run locally

```bash
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

### Retrain the model

Run all cells in `student-placement-predictor.ipynb`; it regenerates `model.pkl`.

## Known Issues & Suggested Improvements

- **Prediction labels may be inverted.** `app.py` shows "Placed" when the model outputs `0` and "Not Placed" when it outputs `1`. In the dataset, label `1` has the higher average CGPA (7.45 vs 6.16), which suggests `1` means placed. Verify what `placed` encodes and fix the mapping if needed.
- **Out-of-range inputs.** The app accepts any IQ above 0, but training IQ values only span 83–121. Predictions for values far outside the training range (such as an IQ of 140) are unreliable. Consider tightening validation to the training ranges.
- **Small dataset** (300 rows) and a single 60-row test split, so the 93.3% figure is noisy. Use cross-validation for a more reliable estimate.
- **Debug mode is on** (`app.run(debug=True)`). Turn it off for any public deployment.
- **Prediction history** in `prediction_history.csv` contains test entries; clear it before sharing.
- The model is trained on unscaled features. SVMs are scale-sensitive, so wrapping scaler and model in a single `Pipeline` may improve results.

## Tech Stack

Python · Flask · scikit-learn · pandas · NumPy · HTML/CSS/JavaScript
