import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

#Data
Data = ["Artificial intelligence is a broad field that encompasses various subfields including machine learning, deep learning, and natural language processing, each of which contributes to the development of intelligent systems capable of solving complex problems such as image recognition, speech processing, and autonomous decision-making."  ]

#Word Indes
Token = Tokenizer()
Token.fit_on_texts(Data)
Seq = Token.texts_to_sequences(Data)
X = []
y = []
for seq in Seq:
    for i in range(1, len(seq)):
        X.append(seq[:i])
        y.append(seq[i])

#RNN
Len = max(len(x) for x in X)
X = np.array([np.pad(x, (Len - len(x), 0), mode='constant') for x in X])
y = np.array(y)
y = to_categorical(y, num_classes=len(Token.word_index) + 1)
Model = Sequential()
Model.add(Embedding(len(Token.word_index) + 1, 10, input_length=Len))
Model.add(SimpleRNN(50, return_sequences=False))
Model.add(Dense(len(Token.word_index) + 1, activation='softmax'))
Model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

#Train
Model.fit(X, y, epochs=50, verbose=2)
#Function
def Predict(Model, Text, Token):
    Sequence = Token.texts_to_sequences([Text])
    Sequence = np.pad(Sequence[0], (Len - len(Sequence[0]), 0), mode='constant')
    Prediction = Model.predict(np.array([Sequence]))
    PredictWord = Token.index_word[np.argmax(Prediction)]
    return PredictWord

Input = "Artificial intelligence is a"
PW = Predict(Model, Input, Token)
print(f"پيش بيني کلمه بعد: {PW}")
