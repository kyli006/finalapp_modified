import streamlit as st
import pandas as pd
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import altair as alt
from sklearn.model_selection import train_test_split
#rubric: scikit-learn/Keras
#https://www.kaggle.com/bayunova/is-this-sentence-completed
st.title("Final-app")
df = pd.read_json("/Users/jiahaoqiu/Documents/UCI 2021 Fall/Math 10/finished_sentences.json")
#create a set of punctuation as filter
exclude = set(string.punctuation)
def rem_punc(text):
    text = ''.join(ch for ch in text if ch not in exclude)
    return text
#remove punctuation and get the max length of sentence 
df["sentence_clean"] = df["sentence"].apply(lambda x: rem_punc(x))
df["is_finished_num"] = [1 if x == 'Finished' else 0 for x in df['is_finished']]
df["sentence_length"] = df["sentence_clean"].apply(lambda x: len(x))
max_length = df["sentence_length"].max()
st.write(df)
def string_to_num(s):
    s_len = len(s)
    sent = s
    tk = tf.keras.preprocessing.text.Tokenizer(lower = True)
    tk.fit_on_texts(sent)
    s_seq = tk.texts_to_sequences(sent)
    s_pad = tf.keras.preprocessing.sequence.pad_sequences(s_seq, maxlen = s_len, padding = 'post')
    return s_pad

# Transform string to numerical value Using tokenizer from tensorflow, this part refer to the sample code in #https://www.kaggle.com/bayunova/is-this-sentence-completed

X, y = (df['sentence_clean'].values, df['is_finished_num'].values)
tk = tf.keras.preprocessing.text.Tokenizer(lower = True)
tk.fit_on_texts(X)
X_seq = tk.texts_to_sequences(X)
X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_seq, maxlen = 100, padding = 'post')
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size = 0.2)

#let user choose epoch
epo = st.slider("Select the epochs of the model",1,20) + 1
#modeling
model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (100,)),
        #keras.layers.Flatten(),
        keras.layers.Dense(16, activation="sigmoid"),
        keras.layers.Dense(8, activation="sigmoid"),
        keras.layers.Dense(2, activation="sigmoid")
    ]
)

model.compile(
    loss="sparse_categorical_crossentropy", 
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy"],
)
history = model.fit(X_train,y_train,epochs=epo,validation_split = 0.2)
df2 = pd.DataFrame({"loss":history.history["loss"],"accuracy":history.history["accuracy"],"val_loss":history.history["val_loss"],"val_accuracy":history.history["val_accuracy"]})
df2['epoch'] = [x for x in range(0,len(df2["loss"]))]

#Showing performance of the model after different epochs
def make_chart(df):
    df3 = df.copy()
    data = df3.reset_index(drop=True).melt('epoch')
    chart = alt.Chart(data).mark_line(clip = True).encode(
        alt.X('epoch', axis=alt.Axis(tickMinStep=1)),
        alt.Y("value"),
        color = 'variable'
    ).properties(
        title = 'Performance of classifing complete sentences'
    )
    return chart

st.altair_chart(make_chart(df2))
predictions = model.predict(X_test).argmax(axis =1)
category = {0:"Unfinished", 1:"finished"}
pred = [x for x in predictions] 
actu = [x for x in y_test]
df1 = pd.DataFrame({"Prediction":pred, "Actual":actu})
df1['correctness'] = (df1['Prediction'] == df1['Actual'])
#a = []
#text = st.text_input("Sentence to predict:")
#a.append(text)
#text_token = string_to_num(a)
#st.write(category[model.predict(text_token).argmax(axis = 1)[0]])

st.write("The coding part about tokenizing string to numeric values refer to the author in #https://www.kaggle.com/bayunova/is-this-sentence-completed")
