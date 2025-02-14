# README

## Heavy Machinery Price Prediction

### Overview
This project predicts the sale prices of heavy machinery using machine learning techniques. The primary model used is a `RandomForestRegressor`, which has been trained and tested on a dataset containing various features, including machine specifications, sale dates, and external farm-related data. The goal is to develop a reliable model that minimizes prediction error and generalizes well to unseen data.

---

### Project Structure
- **Data Preprocessing**:
  - The dataset is preprocessed by extracting date features from the `saledate` column.
  - Categorical variables are converted to numeric values using category encoding.
  - Missing values are handled through imputation (e.g., median for numerical columns, categorical filling with 'Unknown' or 'None or Unspecified').
  - Additional features such as `ValidYearMade` and `SizeCategory` are created.
  - Rare categories are grouped under "Other."
  - Farm-related data (such as `Farms_per_Capita` and `Farms_per_Square_Mile`) is merged based on state and year.
  - Sale prices are adjusted for inflation using CPI (Consumer Price Index) normalization.

- **Feature Engineering**:
  - Target encoding is applied to categorical features such as `fiBaseModel`, `fiSecondaryDesc`, and `fiProductClassDesc`.
  - `SizeCategory` is mapped based on predefined groupings.

- **Model Training**:
  - The dataset is split into training (80%) and testing (20%) subsets.
  - A `RandomForestRegressor` model is trained with the following hyperparameters:
    - `n_estimators=200`
    - `max_features=0.5`
    - `min_samples_leaf=1`
    - `min_samples_split=14`
    - `max_depth=25`
    - `max_samples=None`
    - `random_state=42`
  - The model is evaluated using Root Mean Squared Error (RMSE).

- **Prediction and Output**:
  - The trained model is applied to the validation dataset to predict sale prices.
  - Predictions are stored in a new CSV file with `SalesID` and `SalePrice`.

---

### Dependencies
This project requires the following Python libraries:
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `joblib`
- `warnings`

Install dependencies using:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn joblib
```

---

### Running the Code
1. Load the datasets (`df_train`, `df_valid`, and `farm_data`) from CSV files.
2. Preprocess the training and validation datasets by handling missing values, encoding categorical variables, and adding new features.
3. Train the `RandomForestRegressor` model on the training data.
4. Evaluate the model using RMSE on the training and testing datasets.
5. Generate sale price predictions for the validation dataset.
6. Save predictions to a CSV file.

Run the script with:
```bash
tractor project.ipynb
```

---

### Performance Metrics
- RMSE is calculated for both training and testing datasets to evaluate model performance.
- The model aims to minimize the error gap between training and testing RMSE to avoid overfitting.

---

### Future Improvements
- Experiment with additional feature engineering techniques.
- Tune hyperparameters further using grid search or Bayesian optimization.
- Test alternative models such as `XGBoost` or `GradientBoostingRegressor`.
- Incorporate more external data sources to enhance predictions.

---

### Author
This project was developed as part of a data science course focused on machine learning for predicting machinery prices.

