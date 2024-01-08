#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import tkinter as tk

import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python')


def kiem_tra_text():
    input_text = entry.get()
    # Thực hiện xử lý kiểm tra với input_text ở đây
    # Ví dụ: In kết quả ra terminal
    new_texts = [input_text]
    with open("/Users/daomanh/Desktop/Fake-News-Detection/server/Backendtokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
    new_sequences = tokenizer.texts_to_sequences(new_texts)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=1000, padding='post')

    # Đường dẫn đến tệp tin mô hình .h5
    model_path = "/Users/daomanh/Desktop/Fake-News-Detection/server/BackendFakeNewDetectionCNN16layers.h5"

    # Tải mô hình từ tệp tin
    model = load_model(model_path)
    predictions = model.predict(new_padded_sequences)
    print("Predictions:", predictions)
    if predictions[0][0] > 0.5:
        confidence = predictions[0][0] * 100
        #return f"Thông tin trên là đúng sự thật ({confidence:.2f}%)"
        return "Thông tin trên là đúng sự thật ({:.2f}%)".format(confidence)

    else:
        confidence = (predictions[0][0]) * 100
        #return f"Thông tin trên là sai sự thật ({confidence:.2f}%)"
        return "Thông tin trên là sai sự thật ({:.2f}%)".format(confidence)



# Tạo cửa sổ giao diện
root = tk.Tk()
root.title("Ứng dụng Kiểm tra Text")

# Tạo khung nhập văn bản
entry = tk.Entry(root, width=30)
entry.pack(pady=10)

# Tạo nút kiểm tra và liên kết với hàm kiem_tra_text
button = tk.Button(root, text="Kiểm tra", command=kiem_tra_text)
button.pack()

# Bắt đầu vòng lặp sự kiện
root.mainloop()