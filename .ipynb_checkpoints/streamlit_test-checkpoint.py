import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras.models import load_model

st.set_page_config(page_title="전력소비 예측 시스템", layout="wide")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

@st.cache_data
def load_data():
    df = pd.read_csv("./dataset/KAG_energydata_complete.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["Hour"] = df["date"].dt.hour
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    df["Appliances_lag1"] = df["Appliances"].shift(1)
    df["Appliances_lag2"] = df["Appliances"].shift(2)
    df["Appliances_lag3"] = df["Appliances"].shift(3)

    df = df.dropna().reset_index(drop=True)
    return df

@st.cache_resource
def load_artifacts():
    model = load_model("./model/model.keras", compile=False)
    with open("./model/X_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open("./model/y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    with open("./model/model_config.pkl", "rb") as f:
        config = pickle.load(f)
    return model, x_scaler, y_scaler, config

df = load_data()
model, x_scaler, y_scaler, config = load_artifacts()

features = config["features"]
seq_length = config["seq_length"]

def make_sequence_from_row(row_values, seq_length):
    arr = np.array([row_values], dtype=float)
    seq = np.repeat(arr[np.newaxis, :, :], seq_length, axis=1)
    seq_2d = seq.reshape(-1, seq.shape[2])
    seq_scaled = x_scaler.transform(seq_2d).reshape(seq.shape)
    return seq_scaled

def predict_one(row_values):
    seq_scaled = make_sequence_from_row(row_values, seq_length)
    pred_scaled = model.predict(seq_scaled, verbose=0)
    pred_log = y_scaler.inverse_transform(pred_scaled)
    pred = np.expm1(pred_log)[0][0]
    return pred

def predict_batch(df_input):
    seq_list = []

    for _, row in df_input.iterrows():
        row_values = [
            row["T_out"], row["RH_out"], row["T1"], row["RH_1"], row["lights"],
            row["Tdewpoint"], row["Press_mm_hg"], row["Hour_sin"], row["Hour_cos"],
            row["Appliances_lag1"], row["Appliances_lag2"], row["Appliances_lag3"]
        ]

        arr = np.array([row_values], dtype=float)
        seq = np.repeat(arr[np.newaxis, :, :], seq_length, axis=1)
        seq_list.append(seq[0])

    seq_array = np.array(seq_list)  # (N, seq_length, num_features)

    seq_2d = seq_array.reshape(-1, seq_array.shape[2])
    seq_scaled = x_scaler.transform(seq_2d).reshape(seq_array.shape)

    pred_scaled = model.predict(seq_scaled, verbose=0)
    pred_log = y_scaler.inverse_transform(pred_scaled)
    pred = np.expm1(pred_log).flatten()

    return pred

tab1, tab2 = st.tabs(["전력소비 예측", "전력량 예측 결과"])

with tab1:
    st.subheader("날씨 기반 전력소비 예측")

    selected_date = st.selectbox(
        "일자 선택",
        options=df["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()[:300]
    )

    selected_row = df[df["date"].dt.strftime("%Y-%m-%d %H:%M") == selected_date].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        T_out = st.number_input("실외온도", value=float(selected_row["T_out"]))
        RH_out = st.number_input("실외습도", value=float(selected_row["RH_out"]))
        T1 = st.number_input("실내온도1", value=float(selected_row["T1"]))
        RH_1 = st.number_input("실내습도1", value=float(selected_row["RH_1"]))

    with col2:
        lights = st.number_input("조명전력", value=float(selected_row["lights"]))
        Tdewpoint = st.number_input("이슬점온도", value=float(selected_row["Tdewpoint"]))
        Press_mm_hg = st.number_input("기압", value=float(selected_row["Press_mm_hg"]))
        hour = st.slider("시간", 0, 23, int(selected_row["Hour"]))

    with col3:
        Appliances_lag1 = st.number_input("직전 전력값 1", value=float(selected_row["Appliances_lag1"]))
        Appliances_lag2 = st.number_input("직전 전력값 2", value=float(selected_row["Appliances_lag2"]))
        Appliances_lag3 = st.number_input("직전 전력값 3", value=float(selected_row["Appliances_lag3"]))

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    row_values = [
        T_out, RH_out, T1, RH_1, lights,
        Tdewpoint, Press_mm_hg, hour_sin, hour_cos,
        Appliances_lag1, Appliances_lag2, Appliances_lag3
    ]

    if st.button("예측하기"):
        with st.spinner("예측 중입니다..."):
            pred = predict_one(row_values)

        st.success(f"예상 전력소비량: {pred:.2f} Wh")

        result_df = pd.DataFrame({
            "일자": [selected_date],
            "실외온도": [T_out],
            "실외습도": [RH_out],
            "실내온도1": [T1],
            "실내습도1": [RH_1],
            "조명전력": [lights],
            "예상 전력소비량": [round(pred, 2)]
        })
        st.dataframe(result_df, width="stretch")

with tab2:
    st.subheader("전력량 예측 결과")

    days = st.selectbox("예측 기간", [7, 14, 30], index=0)

    if st.button("기간별 예측하기"):
        with st.spinner("기간별 예측 계산 중입니다..."):
            rows_per_day = 24 * 6

            recent_df = df.tail(days * rows_per_day).copy()

            recent_df = recent_df.iloc[::6].copy()

            pred_list = predict_batch(recent_df)

            recent_df["예측 전력량"] = pred_list

            recent_df["날짜"] = recent_df["date"].dt.date

            daily_result = recent_df.groupby("날짜").agg({
                "Appliances": "mean",
                "예측 전력량": "mean"
            }).reset_index()

            daily_result.columns = ["날짜", "실제 전력량", "예측 전력량(Wh)"]

            # 소수점 정리
            daily_result["실제 전력량"] = daily_result["실제 전력량"].round(2)
            daily_result["예측 전력량(Wh)"] = daily_result["예측 전력량(Wh)"].round(2)

        left, right = st.columns([1, 1])

        with left:
            st.markdown(f"##### {days}일 예측 결과")
            st.dataframe(daily_result, width="stretch")

        with right:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(daily_result["날짜"], daily_result["실제 전력량"], label="실제 전력량", marker="o")
            ax.plot(daily_result["날짜"], daily_result["예측 전력량(Wh)"], label="예측 전력량", marker="o")
            ax.set_title(f"{days}일 실제값과 예측값 비교")
            ax.set_xlabel("날짜")
            ax.set_ylabel("전력량")
            ax.tick_params(axis="x", rotation=45)
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)