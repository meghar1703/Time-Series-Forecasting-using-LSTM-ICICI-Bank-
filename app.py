import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

st.set_page_config(
    page_title="NIFTY50 LSTM Forecasting",
    layout="wide"
)

st.title(" NIFTY-50 Time Series Forecasting using LSTM")

uploaded_file = st.file_uploader(
    "Upload Stock CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    df["Date"] = pd.to_datetime(df["Date"])

    features = [
        'Prev Close',
        'Open',
        'High',
        'Low',
        'Last',
        'Close',
        'VWAP',
        'Volume',
        'Turnover',
        'Trades',
        'Deliverable Volume'
    ]

    data = df[features].values.astype(np.float32)

    scaler = StandardScaler()

    split = int(len(data)*0.8)

    train_data = scaler.fit_transform(data[:split])
    test_data = scaler.transform(data[split:])

    INPUT_STEPS = 15
    OUTPUT_STEPS = 5

    def create_sequences(data):

        X=[]
        y=[]

        for i in range(
            len(data)-INPUT_STEPS-OUTPUT_STEPS
        ):
            X.append(
                data[i:i+INPUT_STEPS]
            )

            y.append(
                data[
                    i+INPUT_STEPS:
                    i+INPUT_STEPS+OUTPUT_STEPS
                ]
            )

        return np.array(X), np.array(y)

    X_test,y_test=create_sequences(test_data)

    X_test=torch.tensor(
        X_test,
        dtype=torch.float32
    )

    class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=11,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(
            64,
            5 * 11
        )

        self.output_steps = 5
        self.num_features = 11

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out.view(
            -1,
            self.output_steps,
            self.num_features
        )

model = RNNModel()

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    with torch.no_grad():

        preds=model(X_test).numpy()

    preds_flat=preds.reshape(-1,11)
    y_flat=y_test.reshape(-1,11)

    inv_preds=scaler.inverse_transform(
        preds_flat
    )

    inv_actuals=scaler.inverse_transform(
        y_flat
    )

    st.subheader("Overall Metrics")

    mse=mean_squared_error(
        inv_actuals,
        inv_preds
    )

    rmse=np.sqrt(mse)

    mae=mean_absolute_error(
        inv_actuals,
        inv_preds
    )

    r2=r2_score(
        inv_actuals,
        inv_preds
    )

    mape=np.mean(
        np.abs(
            (inv_actuals-inv_preds)
            /(inv_actuals+1e-8)
        )
    )*100

    acc=100-mape

    col1,col2,col3=st.columns(3)

    col1.metric("MSE",f"{mse:.2f}")
    col2.metric("RMSE",f"{rmse:.2f}")
    col3.metric("MAE",f"{mae:.2f}")

    col1,col2,col3=st.columns(3)

    col1.metric("R²",f"{r2:.4f}")
    col2.metric("MAPE",f"{mape:.2f}%")
    col3.metric("Accuracy",f"{acc:.2f}%")

    st.subheader("Per Feature Metrics")

    metric_rows=[]

    for i,name in enumerate(features):

        r2_val=r2_score(
            inv_actuals[:,i],
            inv_preds[:,i]
        )

        metric_rows.append(
            [name,r2_val]
        )

    metric_df=pd.DataFrame(
        metric_rows,
        columns=["Feature","R2"]
    )

    st.dataframe(metric_df)

    st.subheader("All Feature Predictions")

    fig,axes=plt.subplots(
        4,
        3,
        figsize=(18,12)
    )

    axes=axes.flatten()

    for i in range(11):

        axes[i].plot(
            inv_actuals[:200,i],
            label="Actual"
        )

        axes[i].plot(
            inv_preds[:200,i],
            label="Predicted"
        )

        axes[i].set_title(
            features[i]
        )

    plt.tight_layout()

    st.pyplot(fig)

    st.subheader(
        "Close Price Forecast"
    )

    close_idx=5

    fig2,ax=plt.subplots(
        figsize=(14,5)
    )

    ax.plot(
        inv_actuals[:,close_idx],
        label="Actual"
    )

    ax.plot(
        inv_preds[:,close_idx],
        label="Predicted"
    )

    ax.legend()

    st.pyplot(fig2)

    pred_df=pd.DataFrame(
        inv_preds,
        columns=features
    )

    csv=pred_df.to_csv(
        index=False
    )

    st.download_button(
        "Download Predictions",
        csv,
        "predictions.csv",
        "text/csv"
    )
