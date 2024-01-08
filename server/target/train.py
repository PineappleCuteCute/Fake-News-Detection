from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Model
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, concatenate, Input


# Đường dẫn đến tệp txt
file_path = "/Users/daomanh/Desktop/Fake-News-Detection/Backend/data/train_sentences.txt"

# Mở tệp và đọc nội dung vào biến
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Tách nội dung thành các phần tử của mảng (ví dụ: tách theo dòng)
lines = content.split('\n')

# Chuyển đổi thành mảng NumPy
train_setences = np.array(lines)
train_setences = np.delete(train_setences, -1)


file_path="/Users/daomanh/Desktop/Fake-News-Detection/Backend/data/test_sentences.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Tách nội dung thành các phần tử của mảng (ví dụ: tách theo dòng)
lines = content.split('\n')

# Chuyển đổi thành mảng NumPy
test_sentences = np.array(lines)
test_sentences = np.delete(test_sentences, -1)
vocab_size = 30000
embedding_dim = 300
max_len = 1000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_setences)

train_sequences = tokenizer.texts_to_sequences(train_setences)
padded_train_sequences = pad_sequences(train_sequences, maxlen = max_len, truncating='post', padding = 'post')

test_sequences = tokenizer.texts_to_sequences(test_sentences)
padded_test_sequences = pad_sequences(test_sequences, maxlen = max_len, truncating='post', padding = 'post')



kernel_size = 3
pool_size = 2
lstm_units = 128
rnn_units = 64
dropout_rate = 0.5
num_filters = 32

# Mô hình CNN - 16 layers theo bài báo "Detection of fake news using deep learning CNN–RNN based methods"
# Link research paper:  https://doi.org/10.1016/j.icte.2021.10.003



# các thông số
vocab_size = 20000   
embedding_dim = 300   
max_len = 1000
filters = 128

# Layer input
inputs = Input(shape=(max_len,))

# Layer embedding
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(inputs)

# Các layer tích chập và pooling với kernel_size khác nhau (3,4,5) - Tổng 6 layers
conv1 = Conv1D(filters=filters, kernel_size=3, activation='relu')(embedding)
pool1 = MaxPooling1D(5)(conv1)

conv2 = Conv1D(filters=filters, kernel_size=4, activation='relu')(embedding)
pool2 = MaxPooling1D(5)(conv2)

conv3 = Conv1D(filters=filters, kernel_size=5, activation='relu')(embedding)
pool3 = MaxPooling1D(5)(conv3)

# Layer ghép các lớp pooling lại với nhau
merged = concatenate([pool1, pool2, pool3], axis = 1)

# 1 Layer tích chập và 1 Layer pooling
x = Conv1D(filters=filters, kernel_size=5, activation='relu')(merged)
x = MaxPooling1D(5)(x)

# 1 Layer tích chập và 1 Layer pooling
x = Conv1D(filters=filters, kernel_size=5, activation='relu')(x)
x = MaxPooling1D(5)(x)

# 1 layer GlobalMaxPooling
x = GlobalMaxPooling1D()(x)

# Layer Fully Connected
x = Dense(128, activation='relu')(x)

# Layer output
outputs = Dense(1, activation='sigmoid')(x)

# Tạo mô hình
model = Model(inputs=inputs, outputs=outputs)

# Biên soạn mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

file_path="/Users/daomanh/Desktop/fakenew/data/train_labels.txt"
train_labels = np.loadtxt(file_path, dtype=int)
file_path="/Users/daomanh/Desktop/fakenew/data/test_labels.txt"
test_labels = np.loadtxt(file_path, dtype=int)

#Huấn luyện mô hình
#history = model.fit(padded_train_sequences, train_labels,batch_size = 512, epochs=5, validation_data=(padded_test_sequences, test_labels))
#model.save("C:/Users/Laptop/PycharmProjects/pythonProject1/FakeNewDetectionCNN16layers.h5")