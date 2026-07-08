# Titanic-Ml-Pipeline-


A baseline end-to-end ML pipeline predicting passenger survival on the Titanic dataset (Kaggle competition), built while working through *Hands-On Machine Learning* (Chapter 2).

## Overview
- Cleaned and preprocessed the Titanic dataset (missing value imputation, categorical encoding)
- Built a `ColumnTransformer` + `Pipeline` for numeric scaling and categorical one-hot encoding
- Trained a Logistic Regression baseline model
- Generated Kaggle-formatted submission

## Results
- Validation accuracy: **81.0%**
- Kaggle public leaderboard score: **0.76794**

## Tech Stack
- Python, Pandas, NumPy
- scikit-learn (`SimpleImputer`, `OneHotEncoder`, `StandardScaler`, `ColumnTransformer`, `Pipeline`, `LogisticRegression`)

## Next Steps
- Feature engineering: extract `Title` from `Name`, `FamilySize` from `SibSp`/`Parch`
- Try `RandomForestClassifier` / `HistGradientBoostingClassifier`
- Cross-validation instead of single train/val split

