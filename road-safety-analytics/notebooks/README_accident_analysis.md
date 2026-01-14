# Road Accident Risk Prediction Analysis

A comprehensive Python script for exploratory data analysis, feature engineering, and machine learning model development to predict road accident risk levels.

## Overview

This script analyzes road accident data to identify patterns, relationships, and risk factors. It performs complete end-to-end machine learning workflow including data exploration, feature engineering, model training, and performance evaluation using multiple classification algorithms.

**What You'll Learn:**
- Exploratory Data Analysis (EDA) techniques
- Feature engineering and encoding strategies
- Model training with multiple algorithms (Random Forest, Gradient Boosting, Logistic Regression)
- Model evaluation with ROC curves, confusion matrices, and classification reports
- Data visualization and correlation analysis

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computations |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical visualization |
| `scikit-learn` | Machine learning models and preprocessing |

## Installation

### Step 1: Clone or Download

```bash
git clone <repository-url>
cd road-accident-analysis
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data Requirements

### Input Dataset: `synthetic_road_accidents_100k.csv`

The script expects a CSV file with the following columns:

| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `road_type` | Categorical | Type of road (highway, urban, rural) | "highway" |
| `lighting` | Categorical | Lighting condition (day, night, twilight) | "night" |
| `weather` | Categorical | Weather condition (clear, rain, snow, fog) | "rain" |
| `time_of_day` | Categorical | Time period (morning, afternoon, evening, night) | "evening" |
| `num_lanes` | Numeric | Number of lanes | 4 |
| `curvature` | Numeric | Road curvature/bend intensity (0-10) | 3.5 |
| `speed_limit` | Numeric | Speed limit in km/h | 80 |
| `accident_risk` | Numeric | Target variable - risk score (0-1) | 0.65 |
| `road_signs_present` | Boolean | Whether road signs are present | True |
| `public_road` | Boolean | Whether it's a public road | True |
| `holiday` | Boolean | Whether it's a holiday | False |
| `school_season` | Boolean | Whether it's school season | True |

### Sample Data Structure

```csv
road_type,lighting,weather,time_of_day,num_lanes,curvature,speed_limit,accident_risk,road_signs_present,public_road,holiday,school_season
highway,day,clear,morning,4,2.1,100,0.35,True,True,False,True
urban,night,rain,evening,2,6.5,50,0.72,True,True,True,False
rural,twilight,fog,night,1,4.2,60,0.58,False,True,False,True
```

## Usage

### Basic Usage

Run the script directly:

```bash
python road_accident_analysis.py
```

### Expected Output

The script generates comprehensive console output with:
- Data shape and basic statistics
- Categorical features distribution
- Risk analysis by road characteristics
- Correlation analysis
- Model performance metrics
- ROC curves and confusion matrices

## Script Structure

The script is organized into **6 main sections**:

### Section 1: Data Loading and Exploration

```python
df = pd.read_csv('synthetic_road_accidents_100k.csv')
```

**Outputs:**
- Dataset shape (rows × columns)
- Data types for each column
- Missing value counts
- Descriptive statistics (mean, std, min, max, etc.)

### Section 2: Exploratory Data Analysis

Analyzes patterns in categorical and numerical features:

**Categorical Analysis:**
- Value counts for each categorical feature
- Risk statistics (mean, median, std) grouped by each feature

**Numerical Analysis:**
- Correlation matrix with accident_risk
- Distribution patterns

**Key Insights Generated:**
- Which lighting conditions have highest accident risk
- Weather impact on accident probability
- Road type risk comparison
- Speed limit correlation with accidents

### Section 3: Feature Engineering

Creates new features for better model performance:

```python
# Binary features from continuous variables
df['high_curvature'] = (df['curvature'] > df['curvature'].median()).astype(int)
df['high_speed'] = (df['speed_limit'] > df['speed_limit'].median()).astype(int)

# Binary target variable
risk_threshold = df['accident_risk'].median()
df['high_risk'] = (df['accident_risk'] > risk_threshold).astype(int)
```

**New Features:**
- `high_curvature`: Binary flag for high road curvature
- `high_speed`: Binary flag for high speed limits
- `high_risk`: Binary target (1 if risk > median, 0 otherwise)

### Section 4: Data Preparation

Prepares data for machine learning:

```python
# Encode categorical variables
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_model[col + '_encoded'] = le.fit_transform(df_model[col])
```

**Processing Steps:**
- Label encode categorical features
- Convert boolean columns to integers
- Separate features (X) and targets (y)
- Create both binary and continuous targets

### Section 5: Train-Test Split and Scaling

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Default Configuration:**
- Test size: 20%
- Train size: 80%
- Random state: 42 (for reproducibility)

### Section 6: Model Training and Evaluation

Trains three classification models:

1. **Logistic Regression** — Baseline model, fast and interpretable
2. **Random Forest** — Ensemble method, handles non-linearity
3. **Gradient Boosting** — Advanced ensemble, typically best performance

**Evaluation Metrics:**
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC-AUC Score
- ROC Curve visualization

## Feature Descriptions

### Categorical Features

| Feature | Values | Impact |
|---------|--------|--------|
| `road_type` | highway, urban, rural | Different safety standards per type |
| `lighting` | day, night, twilight | Visibility affects accident risk |
| `weather` | clear, rain, snow, fog | Grip and visibility conditions |
| `time_of_day` | morning, afternoon, evening, night | Traffic patterns and fatigue |

### Numerical Features

| Feature | Range | Impact |
|---------|-------|--------|
| `num_lanes` | 1-8 | More lanes = capacity, less congestion |
| `curvature` | 0-10 | Higher = more dangerous bends |
| `speed_limit` | 30-120 km/h | Higher speeds = higher accident severity |

### Boolean Features

| Feature | Impact |
|---------|--------|
| `road_signs_present` | Signs provide warnings |
| `public_road` | Public roads have regulations |
| `holiday` | Holiday traffic patterns differ |
| `school_season` | School zones have more activity |

## Models Explained

### Logistic Regression

Simple linear model for classification:
- **Pros:** Fast, interpretable, good baseline
- **Cons:** Assumes linear relationships
- **Use when:** Need fast predictions and model interpretability

### Random Forest

Ensemble of decision trees:
- **Pros:** Handles non-linear patterns, feature importance available
- **Cons:** Slower to train, black-box model
- **Use when:** Non-linear relationships expected

### Gradient Boosting

Sequential ensemble building:
- **Pros:** Often best performance, handles complex patterns
- **Cons:** Slower, harder to interpret, prone to overfitting
- **Use when:** Accuracy is critical

## Interpreting Results

### Confusion Matrix

```
              Predicted Negative  Predicted Positive
Actual Negative    TN (True Neg)    FP (False Pos)
Actual Positive    FN (False Neg)   TP (True Pos)
```

- **TN:** Correctly predicted low-risk roads
- **FP:** Incorrectly flagged low-risk as high-risk
- **FN:** Missed actual high-risk roads
- **TP:** Correctly identified high-risk roads

### ROC-AUC Score

Ranges 0-1:
- **0.5:** Random guessing
- **0.7-0.8:** Good model
- **0.8-0.9:** Very good model
- **0.9+:** Excellent model

### Classification Metrics

- **Precision:** Of predicted high-risk, how many actually are
- **Recall:** Of actual high-risk, how many did we find
- **F1-Score:** Harmonic mean of precision and recall
- **Support:** Number of samples in each class

## Customization

### Change Test-Train Split

```python
# Default: 80-20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42  # Change to 30%
)
```

### Change Risk Threshold

```python
# Default: median
risk_threshold = df['accident_risk'].quantile(0.75)  # Top 25% as high-risk
df['high_risk'] = (df['accident_risk'] > risk_threshold).astype(int)
```

### Add More Features

```python
# Create interaction feature
df['high_speed_night'] = df['high_speed'] * (df['lighting'] == 'night').astype(int)

# Add to feature list
feature_cols.append('high_speed_night')
```

### Train Different Models

```python
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Support Vector Machine
svm_model = SVC(probability=True)
svm_model.fit(X_train_scaled, y_train)

# Neural Network
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
nn_model.fit(X_train_scaled, y_train)
```

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | CSV file not in correct location | Place `synthetic_road_accidents_100k.csv` in script directory |
| `KeyError: column name` | CSV missing expected column | Verify all required columns are present |
| `Low model accuracy` | Poor feature quality or imbalanced data | Try different features or adjust risk threshold |
| `Memory error` | Dataset too large | Use `nrows` parameter in `pd.read_csv()` |

## Tips for Better Results

1. **Data Quality** — Ensure input CSV has no missing values in critical columns
2. **Feature Engineering** — Create domain-specific features (e.g., rush_hour, weather_changes)
3. **Class Imbalance** — If high-risk roads are rare, use `class_weight='balanced'` in models
4. **Cross-Validation** — Use k-fold cross-validation for more reliable metrics
5. **Hyperparameter Tuning** — Use GridSearchCV to find optimal parameters
6. **Feature Scaling** — Already done, but verify if adding new features
7. **Model Comparison** — Test multiple models to find best fit
8. **Ensemble Methods** — Combine predictions from multiple models

## Advanced Usage

### Using Different Scoring Metrics

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate specific metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf_model, X_scaled, y_binary, cv=5)
print(f"Average CV Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Feature Importance

```python
# For Random Forest
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), params, cv=5)
grid_search.fit(X_train_scaled, y_train)
print(f"Best parameters: {grid_search.best_params_}")
```

## Output Files

The script generates console output and can be extended to save:

- **Trained models** — Save with `joblib.dump(model, 'model.pkl')`
- **Predictions** — Export as CSV for further analysis
- **Visualizations** — Save plots with `plt.savefig()`
- **Reports** — Generate PDF reports with model metrics

Example:

```python
import joblib

# Save model
joblib.dump(rf_model, 'accident_risk_model.pkl')

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'probability': y_pred_proba[:, 1]
})
predictions_df.to_csv('predictions.csv', index=False)
```

## Performance Benchmarks

Expected model performance (varies with data):

| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|----------|
| Logistic Regression | ~65-70% | ~0.70 | ~0.65 |
| Random Forest | ~75-80% | ~0.80 | ~0.76 |
| Gradient Boosting | ~78-82% | ~0.82 | ~0.79 |

## Next Steps

1. **Deploy Model** — Save best model and use in production
2. **Continuous Learning** — Retrain with new accident data regularly
3. **Feature Expansion** — Add weather severity, traffic volume, vehicle type
4. **Real-time Prediction** — Integrate with traffic monitoring systems
5. **Visualization Dashboard** — Create interactive dashboard for insights
6. **Business Integration** — Use predictions for road safety campaigns

## References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [ROC Curve Explanation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)

## License

This project is provided for educational purposes. Modify and distribute as needed.

---

**Version:** 1.0  
**Last Updated:** January 2024  
**Python:** 3.8+  
**Maintainer:** Your Team