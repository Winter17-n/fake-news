# # -------------importing Libraries.--------------
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.layers import SimpleRNN , Dense , Embedding , LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

# ---------Load data------------
Fake = pd.read_csv("Fake.csv")
Real = pd.read_csv("True.csv")

#-----------Add labels-----------
Fake['label'] = 'Fake News.'
Real['label'] = 'Real News.'

# print(Fake)

#------------Combine datasets----------
data = pd.concat([Fake, Real])
data = data[['text', 'label','date']]

# data['date'] = pd.to_datetime(data['date'], errors = 'coerce')
# data['year'] = data['date'].dt.year 

# year_counts = data['year'].value_counts().sort_index()
# print(data['year'])

# plt.figure(figsize=(10,5))
# plt.bar(year_counts.index, year_counts.values)
# plt.xlabel("Year")
# plt.ylabel("Number of Articles")
# plt.title("Number of News Articles per Year")
# plt.xticks(rotation=45)
# plt.show()

# # --------------Dividing the Data-------------
# X = data['text']
# y = data['label']

# # -----------------Train Test Split----------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # print(X_train.shape)
# # print(X_test.shape)

# #--------------Vectorize text-------------
# vect = TfidfVectorizer(stop_words='english', max_df=0.7)
# tfidf_train = vect.fit_transform(X_train)
# tfidf_test = vect.transform(X_test)

# #--------------Train classifier with logistic Regression--------------
# logistic_model = LogisticRegression(max_iter=1000)
# logistic_model.fit(tfidf_train, y_train)

# # ---------------Training with Naive Bayes---------------
# Naive_model = MultinomialNB()
# Naive_model.fit(tfidf_train,y_train)

# # -----------------Training with SVC(more specifically LiinearSVC)--------------
# svc_model = LinearSVC()                     # used LinearSVC cause it do good wth Text
# svc_model.fit(tfidf_train,y_train)

# # ---------------Predicting and evaluating (For Logistic Regression)-----------------
# Logistic_pred = logistic_model.predict(tfidf_test)
# Logistic_score = accuracy_score(y_test, Logistic_pred)

# # ------------------(For Naive Bayes)----------------
# Naive_pred = Naive_model.predict(tfidf_test) 
# Naive_score = accuracy_score(y_test, Naive_pred)

# # ----------------(For SVC)---------------
# svc_pred= svc_model.predict(tfidf_test)
# svc_score = accuracy_score(y_test,svc_pred)

# # ---------------Printing Values---------------
# print("\nPrediction for Logistic Regression:\n",Logistic_pred)
# print("Prediction for Naive Bayes:\n",Naive_pred)
# print("Prediction for SVC:\n",svc_model)

# # ---------------Checking Accuracy------------
# print(f"\nAccuracy for Logistic Regression : {round(Logistic_score * 100, 2)}%")
# print(f"Accuracy for Naive Bayes : {round(Naive_score * 100, 2)}%")
# print(f"Accuracy for SVC : {round(svc_score *100,2)}%")


# # --------------The Confusion Matrix--------------
# print("\nConfusion Matrix for Logistic Regression: \n",confusion_matrix(y_test, Logistic_pred))
# print("\nConfusion Matrix for Naive Bayes: \n",confusion_matrix(y_test,Naive_pred))
# print("\nConfusion Matrix for SVC: \n",confusion_matrix(y_test,svc_pred))

# # # # -----------Predicting New Values-----------
# new = pd.DataFrame({
#     'text': ['The Indian Space Research Organisation (ISRO) successfully launched its latest communication satellite from the Satish Dhawan Space Centre on Friday. The mission aims to improve broadband connectivity across rural and remote regions of India.']
# })

# # -------------Vectorizing new data----------------
# new_dataf= vect.transform(new['text'])

# # --------------Feeding New Data----------------
# log=logistic_model.predict(new_dataf)

# naive = Naive_model.predict(new_dataf)

# svc = svc_model.predict(new_dataf)

# # -------------Printing New Prediction--------------
# print("\nPrediction Using Logistic Regression:",log)
# print("Prediction using Naive Bayes:",naive)
# print("Prediction using SVC:",svc)

# # ----------importing in the pickle-------------
# import joblib

# #------------Saving Svr model------------
# joblib.dump(svc_model, "svr_model.pkl")

# #------------Saving TF-IDF Vectorizer------
# joblib.dump(vect, "vect.pkl")
# print("model saved")

# # -------------XGBoost--------          # we didn't put it above cause xgboost take labels only as 0 or 1.

# --------------Train/test split-------------
X = data['text']
y = data['label']
le = LabelEncoder()
y = le.fit_transform(data["label"])  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# vect = TfidfVectorizer(stop_words='english', max_df=0.7)

# tfidf_train = vect.fit_transform(X_train)
# tfidf_test = vect.transform(X_test)

# new = pd.DataFrame({
#     'text': ['WASHINGTON (Reuters) - Legislation to provide $81 billion in new disaster aid for U.S. states, Puerto Rico and the U.S. Virgin Islands was put on hold by the Senate on Thursday amid attacks from both Republicans and Democrats. The Republican-controlled House of Representatives passed the legislation earlier on Thursday to help recovery efforts stemming from hurricanes and wildfires. But the Senate put off a vote until at least January, according to some lawmakers and aides, after Democrats complained Puerto Rico was not getting enough help and some fiscal hawks fretted about the overall cost. ']
# })

# # -------------Vectorizing new data----------------
# new_dataf= vect.transform(new['text'])

# mod = xgb.XGBClassifier()
# mod.fit(tfidf_train,y_train)

# predi = mod.predict(tfidf_test)
# sco = accuracy_score(y_test,predi)

# print(f"Accuracy for xg : {round(sco *100,2)}%")

# nw = mod.predict(new_dataf)

# if nw == 1:
#     print("Real News.")
# else:
#     print("Fake News.")
# # print(nw)


# # ----------------RNN(LSTM) Model-----------------  # Didn't put above cause again it take labels as 0 or 1 and it uses tokenizer.
max_words = 5000
max_len = 300

#------------Tokenization------------------------
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq  = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad  = pad_sequences(X_test_seq, maxlen=max_len)

# ---------------- LSTM Model ----------------
Lstm_model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

Lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Lstm_model.summary()

# Training
Lstm_model.fit(
    X_train_pad,
    y_train,
    epochs=7,
    batch_size=64,
    validation_split=0.2
)

# Evaluation
lstm_loss, lstm_acc = Lstm_model.evaluate(X_test_pad, y_test)
print(f"LSTM Accuracy: {round(lstm_acc*100, 2)}%")

def predict(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=max_len)
    prediction = Lstm_model.predict(padded)

    if prediction[0][0] > 0.5:
        return "Real News."
    else:
        return "Fake News.  "

sentence = "WASHINGTON (Reuters) - Legislation to provide $81 billion in new disaster aid for U.S. states, Puerto Rico and the U.S. Virgin Islands was put on hold by the Senate on Thursday amid attacks from both Republicans and Democrats. The Republican-controlled House of Representatives passed the legislation earlier on Thursday to help recovery efforts stemming from hurricanes and wildfires. But the Senate put off a vote until at least January, according to some lawmakers and aides, after Democrats complained Puerto Rico was not getting enough help and some fiscal hawks fretted about the overall cost."

print(predict(sentence))