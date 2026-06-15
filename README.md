# Time-Series-Forecasting-using-LSTM-ICICI-Bank-

## Project Overview

This project implements a multi-output time series forecasting model using LSTM networks to predict stock market behaviour from the NIFTY-50 dataset(ICICI Bank data).

The model learns patterns from historical stock data and forecasts multiple financial features for the next 5 days based on past observations.


## Dataset

* **Source:** NIFTY-50 Stock Market Data (2000–2021), ICICI Bank dataset
* **Provider:** https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data
* **Type:** Multivariate time series

### Features Used

* Price-based:

  * Prev Close, Open, High, Low, Last, Close, VWAP
* Volume-based:

  * Volume, Turnover, Trades, Deliverable Volume

---

## Problem Statement

Given historical stock data:

* **Input:** Last *N* days (e.g., 20 days × 11 features)
* **Output:** Next 5 days (all 11 features)

This is a **multi-step, multi-output forecasting problem**, which is more complex than standard single-value prediction.

---

## Model Architecture

* Model: **LSTM (Recurrent Neural Network)**
* Input shape: `(batch, sequence_length, features)`
* Hidden size: 64
* Output: `5 × 11` predictions

### Key Design Choices

* Uses **last timestep output** for prediction
* Fully connected layer maps hidden state → multi-step output
* Supports sequence-to-sequence forecasting

---

## Data Preprocessing Pipeline

### ✔ Data Cleaning

* Removed invalid values (`inf`, `-inf`)
* Filled missing values using:

  * Linear interpolation
  * Forward fill & backward fill

### Feature Scaling

* Applied **log transformation** to skewed features:

  * Volume, Turnover, Trades, Deliverable Volume
* Standardized all features using `StandardScaler`

### Sequence Generation

* Sliding window approach:

  * Input window: 20 timesteps
  * Output window: 5 timesteps

---

## Model Enhancements

### Weighted Loss Function

* Assigned lower weights to noisy features (volume-related)
* Prevented domination of large-scale features in training

###  Dropout Regularization

* Added dropout to improve generalization

###  Early Stopping

* Stops training when validation loss stops improving
* Prevents overfitting

---

## valuation Metrics

The model is evaluated using:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score

### Evaluation Strategy

* **Feature-wise evaluation** to handle scale differences
* Log-scale evaluation for volume features
* Original-scale evaluation for price features

---

##  Results
<img width="1823" height="1030" alt="lstm3" src="https://github.com/user-attachments/assets/11fee973-b67f-4ca6-a823-38b9cb27f72f" />

<img width="1840" height="1229" alt="lstm2" src="https://github.com/user-attachments/assets/432217e1-2d30-4277-a524-081058fa1d92" />

<img width="1883" height="1271" alt="lstm1" src="https://github.com/user-attachments/assets/1435e921-bdc4-47f4-b922-8c747bf277b7" />


### Overall Performance

* **MSE:** ~261
* **RMSE:** ~13
* **MAE:** ~9.85
* **R² Score:** ~0.50

### Close Price Performance (Key Metric)

* **R² Score:** ~0.95
* **RMSE:** ~20.8

---

## Key Insights

* Price-based features (Open, High, Low, Close) show **strong temporal dependencies**
* Volume-related features are:

  * Highly volatile
  * Less predictable
* Applying **log transformation + weighted loss** significantly improved model stability
* Multi-output forecasting is more challenging than single-target prediction

---

## Visualization

The project includes:

* Feature-wise prediction vs actual plots (subplots)
* Date-wise Close price comparison
* Future 5-day prediction graph

---

## Future Predictions

The model predicts **next 5 days of stock behavior**, including:

* Price trends
* Volume estimates
* Market activity indicators

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## Project Structure

```
├── notebook.ipynb        # Main implementation
├── README.md             # Project documentation
```
---

## Conclusion

This project demonstrates how deep learning models like LSTM can effectively model financial time series data, especially for price prediction.

While predicting market volume remains challenging, the model achieves **high accuracy on key financial indicators**, making it a strong baseline for further research.

---

##  Future Work

* Implement **GRU and compare performance**
* Add **Attention Mechanism**
* Use **Transformer-based models**
* Hyperparameter tuning
* Deploy as a web app
## ⭐ If you found this useful

Consider giving this repository a star!
