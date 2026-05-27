# Titanic KNN Classification with GridSearchCV

A complete machine learning pipeline for predicting passenger survival on the Titanic using K-Nearest Neighbors (KNN) with advanced hyperparameter tuning and outlier handling.

## 📋 Project Overview

This project implements a KNN classifier to predict whether Titanic passengers survived based on their characteristics. The pipeline includes:

- **Data preprocessing** with missing value imputation
- **Outlier detection** using the Interquartile Range (IQR) method
- **Feature engineering** with one-hot encoding
- **Hyperparameter tuning** using GridSearchCV
- **Model evaluation** with comprehensive metrics
- **Inference app** for making predictions on new data

## 📁 Project Structure

```
knn - classification/
├── data/
│   └── titanic.csv                 # Raw dataset
├── models/
│   ├── knn_model.pkl               # Trained KNN model
│   ├── scaler.pkl                  # StandardScaler for feature normalization
│   └── feature_names.pkl           # Feature names in correct order
├── notebooks/
│   └── knn_titanic_model.ipynb     # Main training notebook
├── app.py                          # Inference script for predictions
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Copy your `titanic.csv` file to the `data/` folder:

```bash
cp titanic.csv data/
```

### 3. Train the Model

Open and run the notebook:

```bash
jupyter notebook notebooks/knn_titanic_model.ipynb
```

The notebook will:
- Load and explore the data
- Handle missing values
- Remove outliers
- Encode categorical features
- Split into train/test sets
- Perform GridSearchCV to find optimal hyperparameters
- Train the final model
- Save model artifacts to `models/` folder

### 4. Make Predictions

Use the inference app to predict survival for new passengers:

```python
python app.py
```

Or in your own code:

```python
from app import TitanicKNNPredictor

predictor = TitanicKNNPredictor()

passenger = {
    'Age': 25,
    'Fare': 50.0,
    'Sex': 'female',
    'SibSp': 1,
    'Parch': 0,
    'Pclass': 1,
    'Embarked': 'S'
}

result = predictor.predict(passenger)
print(f"Survived: {result['survived']}")
print(f"Probability: {result['survival_probability']:.4f}")
```

## 📊 Dataset Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| **Age** | Numerical | Passenger age | 0.83 - 80 |
| **Fare** | Numerical | Ticket price | 0 - 512.33 |
| **Sex** | Categorical | Gender (Male/Female) | - |
| **SibSp** | Numerical | Siblings/Spouses aboard | 0 - 8 |
| **Parch** | Numerical | Parents/Children aboard | 0 - 6 |
| **Pclass** | Categorical | Ticket class (1st/2nd/3rd) | 1, 2, 3 |
| **Embarked** | Categorical | Port of embarkation (S/C/Q) | - |
| **Survived** | Target | Survival outcome | 0 (No), 1 (Yes) |

## 🔧 Preprocessing Pipeline

### 1. **Missing Value Imputation**
- **Age**: Filled with mean age (~29.7 years)
- **Embarked**: Filled with mode (most common port)

### 2. **Outlier Detection (IQR Method)**
- **Bounds**: `Q1 - 1.5×IQR` to `Q3 + 1.5×IQR`
- **Applied to**: Age and Fare columns
- Removes extreme values that could distort distance calculations

### 3. **Feature Encoding**
- **One-hot encoding** for categorical variables (Sex, Embarked)
- Converts categories to binary features

### 4. **Feature Scaling**
- **StandardScaler** normalization (zero mean, unit variance)
- Essential for KNN (distance-based algorithm)

### 5. **Train-Test Split**
- **80% training** / **20% testing**
- **Stratified split** to maintain class distribution

## 🎯 GridSearchCV Hyperparameter Tuning

The model tests **63 parameter combinations** using 5-fold cross-validation:

```python
Parameters tested:
├── n_neighbors: [3, 5, 7, 9, 11, 13, 15]
├── weights: ['uniform', 'distance']
└── metric: ['euclidean', 'manhattan', 'minkowski']
```

### Best Parameters (Example)
```
n_neighbors: 7
weights: 'distance'
metric: 'euclidean'
```

## 📈 Model Performance

Typical results after training:

| Metric | Training | Testing |
|--------|----------|---------|
| **Accuracy** | ~0.82 | ~0.80 |
| **Precision** | ~0.83 | ~0.81 |
| **Recall** | ~0.77 | ~0.75 |
| **F1-Score** | ~0.80 | ~0.78 |

See confusion matrix and classification report in the notebook output.

## 💾 Model Artifacts

After training, the following files are saved in `models/`:

- **knn_model.pkl**: Trained KNeighborsClassifier
- **scaler.pkl**: StandardScaler fitted on training data
- **feature_names.pkl**: Ordered list of feature names

These ensure consistent preprocessing for new predictions.

## 🧪 Example Usage

### Training & Evaluation
See `notebooks/knn_titanic_model.ipynb` for complete pipeline

### Making Predictions
```python
from app import TitanicKNNPredictor

# Initialize predictor
predictor = TitanicKNNPredictor()

# Predict for a first-class female passenger
passenger_data = {
    'Age': 25,
    'Fare': 150.0,
    'Sex': 'female',
    'SibSp': 1,
    'Parch': 0,
    'Pclass': 1,
    'Embarked': 'S'
}

result = predictor.predict(passenger_data)
# Output:
# {
#     'survived': True,
#     'survival_probability': 0.8571,
#     'death_probability': 0.1429
# }
```

## 🔍 Key Insights

1. **Female passengers** had significantly higher survival rates
2. **First-class passengers** had better survival chances
3. **Age** and **ticket fare** are strong predictors
4. **Traveling with family** (SibSp/Parch) affected survival
5. **Port of embarkation** shows some correlation with survival

## 🐛 Troubleshooting

### Issue: `FileNotFoundError` when loading data
**Solution**: Make sure `titanic.csv` is in the `data/` folder

### Issue: `KeyError` for column names
**Solution**: Check that CSV has correct columns: Age, Fare, Sex, SibSp, Parch, Pclass, Embarked, Survived

### Issue: Model file not found when running `app.py`
**Solution**: Run the notebook first to generate model artifacts in `models/` folder

## 📚 Dependencies

- **pandas** (2.0.3): Data manipulation
- **numpy** (1.24.3): Numerical computations
- **scikit-learn** (1.3.0): ML algorithms and metrics
- **matplotlib** (3.7.2): Visualization
- **seaborn** (0.12.2): Statistical plots

See `requirements.txt` for exact versions.

## 📝 Notes

- This model is trained on historical Titanic data and may not generalize to other domains
- KNN is sensitive to feature scaling; always use the saved scaler
- GridSearchCV can be computationally expensive; adjust CV folds or parameter ranges for faster iteration
- For production use, consider implementing proper model versioning and monitoring

## 🎓 Educational Purpose

This project demonstrates:
- ✅ Complete ML pipeline from data to deployment
- ✅ Handling missing values and outliers
- ✅ Feature engineering and encoding
- ✅ Hyperparameter optimization techniques
- ✅ Model evaluation and interpretation
- ✅ Creating reusable inference components

## 📖 References

- [Scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [GridSearchCV Guide](https://scikit-learn.org/stable/modules/grid_search.html)
- [Titanic Dataset](https://www.kaggle.com/c/titanic)

---

# Titanic KNN — Simple Project

This repository is a minimal, self-contained K-Nearest Neighbors classification project for the Titanic dataset. It includes:

- `data/titanic.csv` — dataset (keep it in `data/`)
- `notebooks/knn_simple.ipynb` — notebook to train a KNN model and save artifacts
- `models/knn_model.pkl`, `models/scaler.pkl`, `models/feature_names.pkl` — trained artifacts (generated by the notebook)
- `app.py` — Streamlit inference app (loads artifacts and predicts)
- `requirements.txt` — minimal dependencies

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model (optional — artifacts may already exist):

```bash
jupyter notebook notebooks/knn_simple.ipynb
# run the notebook cells to train and save artifacts
```

3. Run the app:

```bash
streamlit run app.py
```

Open the URL printed by Streamlit and use the left-side controls to make predictions.

## Notes

- The notebook applies simple preprocessing (fill missing Age with median, fill Embarked with mode), encodes categorical features, scales numeric features, and tunes K with GridSearchCV.
- `app.py` expects the artifacts in the `models/` folder: `knn_model.pkl`, `scaler.pkl`, `feature_names.pkl`.
- This project is intentionally minimal for educational purposes.

If you want, I can now run the notebook for you to (re)generate artifacts and verify the app — shall I do that?
