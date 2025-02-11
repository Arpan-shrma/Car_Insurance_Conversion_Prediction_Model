# 🚗 Car Insurance Conversion Prediction Model

## 📊 Project Overview
A machine learning project to predict whether customers will purchase car insurance, implementing and comparing Logistic Regression and XGBoost models.

### 💰 Business Constraints
- Revenue: $5.50 for each True Positive
- Cost: $1 for each Positive prediction

## 🎯 Project Goals
- Predict customer insurance purchase decisions (binary classification)
- Maximize revenue while managing costs
- Compare performance of different modeling approaches

## 📦 Dependencies
```python
# Core Data Science
pandas
numpy
scikit-learn
xgboost

# Visualization
seaborn
matplotlib

# Machine Learning
statsmodels
imbalanced-learn
```

## 🔧 Data Processing Steps
1. **Data Cleaning**
   - Handled missing values
   - Removed duplicates
   - Fixed data type inconsistencies

2. **Feature Engineering**
   - Created age-related features
   - Location-based feature binning
   - Vehicle-related feature transformations

3. **Preprocessing**
   - Standard scaling of numeric features
   - SMOTE for class imbalance
   - One-hot encoding for categorical variables

## 🤖 Models Implemented

### Logistic Regression
- Elastic Net regularization
- Feature selection through L1 regularization
- Probability calibration

**Performance:**
- ROC-AUC: 0.644
- Average Precision: 0.611
- Revenue per 100 predictions: $13

### XGBoost
- Hyperparameter tuning via RandomizedSearchCV
- Class weight balancing
- Custom threshold optimization

**Performance:**
- ROC-AUC: 0.652
- Average Precision: 0.623
- Revenue per 100 predictions: $18

## 📈 Model Comparison
![image](https://github.com/user-attachments/assets/a634897e-0e31-45e4-8ee1-774697a24232)

**Key Findings:**
- XGBoost outperformed Logistic Regression in both AUC and revenue
- Both models showed reasonable discrimination ability
- Models achieved optimal performance at different threshold points

## 💡 Business Impact
1. **Revenue Generation**
   - XGBoost: 18 cents per person
   - Logistic Regression: 13 cents per person

2. **Model Efficiency**
   - True Positive Rate: ~60%
   - False Positive Rate: ~40%
   - Cost-effective prediction strategy

## 🚀 Usage
```python
# Example prediction code
import xgboost as xgb

# Load model
model = xgb.XGBClassifier()
model.load_model('model.json')

# Make predictions
predictions = model.predict_proba(X_test)
```

## 📊 Feature Importance
Top influential features:
1. Customer age
2. Vehicle age
3. Years licensed
4. Annual kilometers driven
5. Type of vehicle usage

## 🔍 Model Evaluation
- Validation strategy: 80/20 train-test split
- Metrics: ROC-AUC, Average Precision, Revenue
- Custom revenue-based optimization

## 📝 Authors
- Arpan Sharma
- Harsh Tiwari

## 🎓 Academic Context
Submitted to University of Guelph in partial fulfillment of DATA*6100 requirements.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
