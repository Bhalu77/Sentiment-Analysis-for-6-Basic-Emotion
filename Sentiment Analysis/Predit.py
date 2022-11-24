import pandas as pd
import joblib
import numpy as np

model = joblib.load('Ml_Pipeline')
test = "i am fine"
test = pd.DataFrame([test], columns= ['text'])
test = test['text'].values.astype('U')
py_pred = model.predict_proba(test)
a = np.argmax(py_pred)
if a == 0:
    text = "Relaxed"
    print(text)

if a == 1:
    if py_pred[0][a] < 0.5:
        text = "Slight Anger"
        print(text)
        

    if py_pred[0][a] >= 0.5 and py_pred[0][a] < 0.75:
        text = "Angry"
        print(text)
        

    if py_pred[0][a] >= 0.75:
        text = "Extreme Anger"
        print(text)
        

if a == 2:
    if py_pred[0][a] < 0.5:
        text = "Slight fear"
        print(text)
        
    if py_pred[0][a] >= 0.5 and py_pred[0][a] < 0.75:
        text = "fear"
        print(text)
        

    if py_pred[0][a] >= 0.75:
        text = "Extreme Fear"
        print(text)
        

if a == 3:
    if py_pred[0][a] < 0.5:
        text = "Slightly happy"
        print(text)
        

    if py_pred[0][a] >= 0.5 and py_pred[0][a] < 0.75:
        text = "Happy"
        print(text)
        

    if py_pred[0][a] >= 0.75:
        text = "Extremely happy"
        print(text)
        

if a == 4:
    if py_pred[0][a] < 0.5:
        text = "Slightly Sad"
        print(text)
        

    if py_pred[0][a] >= 0.5 and py_pred[0][a] < 0.75:
        text = "Sad"
        print(text)
        

    if py_pred[0][a] >= 0.75:
        text = "Extremely Sad"
        print(text)
        

if a == 5:
    if py_pred[0][a] < 0.5:
        text = "Slight Surprise"
        print(text)
        

    if py_pred[0][a] >= 0.5 and py_pred[0][a] < 0.75:
        text = "Surprise"
        print(text)
        

    if py_pred[0][a] >= 0.75:
        text = "Extreme Surprised"
        print(text)
