# LinearRegression.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Linear Regression Analysis")

st.sidebar.header("Model Parameter Settings")

# 1. 添加互動控件

# 用戶可以選擇斜率，範圍從 -100 到 100
slope = st.sidebar.slider("Select Slope", min_value=-100.0, max_value=100.0, value=1.0, step=0.1)

# 用戶可以選擇噪聲比例
noise_scale = st.sidebar.slider("Select Noise Scale", min_value=0.0, max_value=100.0, value=10.0, step=1.0)

# 用戶可以選擇數據點數量
num_points = st.sidebar.number_input("Select Number of Points", min_value=10, max_value=1000, value=100, step=10)

# 2. 生成合成數據並訓練模型

@st.cache_data
def generate_data(slope, noise_scale, num_points):
    np.random.seed(42)  # 固定隨機種子以便重現結果
    X = np.random.rand(num_points, 1) * 100  # 生成 [0, 100) 範圍內的隨機數據
    noise = np.random.randn(num_points, 1) * noise_scale  # 添加噪聲
    y = slope * X + noise  # 線性關係
    return X, y

X, y = generate_data(slope, noise_scale, num_points)

# 將數據轉換為 DataFrame 以便顯示
data = pd.DataFrame(np.hstack((X, y)), columns=["X", "y"])

# 3. 顯示部分數據
if st.checkbox("Show Data Sample"):
    st.write("### Data Sample")
    st.write(data.head())

# 4. 建立並訓練線性回歸模型
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 5. 評估模型性能
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.write("## Model Performance Evaluation")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared (R²): {r2:.2f}")

# 6. 繪製實際值與預測值的對比圖

st.write("## Actual Values vs Predicted Values")

fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', edgecolor='k', alpha=0.7, label='Actual Values')
ax.plot(X, y_pred, color='red', linewidth=2, label='Predicted Values')
ax.set_xlabel('Independent Variable X')
ax.set_ylabel('Dependent Variable y')
ax.set_title('Actual Values vs Predicted Values')
ax.legend()

st.pyplot(fig)
