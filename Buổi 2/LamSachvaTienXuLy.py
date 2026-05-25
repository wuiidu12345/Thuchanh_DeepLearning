import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. đọc dữ liệu đã làm sạch
df = pd.read_csv("dulieuxettuyendaihoc_clean.csv")

# 2. tách các tính năng đầu vào (X) và mục tiêu cần dự đoán (y)
# dự đoán điểm DH1 dựa trên điểm số cấp 3 (loại bỏ các cột điểm đại học khỏi X)
X = df.drop(columns=["DH1", "DH2", "DH3"])
y = df["DH1"]

# 3. chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. thiết kế cấu trúc mạng thần kinh
model = Sequential(
    [
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(
            1
        ),  # lớp đầu ra có 1 nút và không dùng activation để dự đoán giá trị liên tục
    ]
)

# 5. cấu hình mô hình
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 6. huấn luyện mô hình
print("bắt đầu huấn luyện...")
history = model.fit(
    X_train, y_train, epochs=60, batch_size=8, validation_split=0.1, verbose=1
)

# 7. đánh giá mô hình trên tập kiểm tra
print("\nđánh giá mô hình trên tập test:")
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"sai số tuyệt đối trung bình (MAE): {mae:.4f}")
