import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
import numpy as np

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


df_fake = pd.read_csv("/Users/daomanh/Desktop/Fake-News-Detection/server/dataset/True.csv")
df_true = pd.read_csv("/Users/daomanh/Desktop/Fake-News-Detection/server/dataset/Fake.csv")


df_fake["class"] = 0
df_true["class"] = 1

df_merge = pd.concat([df_fake, df_true], axis=0)

df = df_merge.drop(["title", "subject", "date"], axis=1)
df = df.sample(frac=1)

df.reset_index(inplace=True)

df.drop(["index"], axis=1, inplace=True)
df["text"] = df["text"].apply(wordopt)
x = df["text"]
y = df["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
train_sentences = x_train.to_numpy()

train_labels = y_train.to_numpy()
print(len(train_sentences))
print(train_sentences[31427])
test_sentences = x_test.to_numpy()
test_labels = y_test.to_numpy()
int32 = np.int32
train_labels = train_labels.astype(int32)
test_labels = test_labels.astype(int32)


# Lưu mảng vào tệp
np.savetxt("/Users/daomanh/Desktop/Fake-News-Detection/server/data/train_sentences.txt", train_sentences, fmt="%s", encoding='utf-8')
np.savetxt("/Users/daomanh/Desktop/Fake-News-Detection/server/data/test_sentences.txt", test_sentences, fmt="%s", encoding='utf-8')
np.savetxt("/Users/daomanh/Desktop/Fake-News-Detection/server/data/train_labels.txt", train_labels, fmt="%d", encoding='utf-8')
np.savetxt("/Users/daomanh/Desktop/Fake-News-Detection/server/data/test_labels.txt", train_labels, fmt="%d", encoding='utf-8')


