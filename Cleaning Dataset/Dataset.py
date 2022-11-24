import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import cleaning

cols = ["texts", "Features"]
df = pd.read_csv(r"Sentiment Analysis/Sentiment_Dataset.csv", names=cols)
X = df['texts']
y = df['Features']

texts = []
for sen in X:
    sen = cleaning.token.tokenize(sen)
    sen = cleaning.remove_noise(sen)
    texts.append(" ".join(sen))

a = []
for i in range(len(texts)):
    if texts[i] == "":
        a.append(i)

f_csv = pd.DataFrame(zip(texts, y))
f_csv = f_csv.drop(labels=a , axis=0)
f_csv.to_csv("FileName.csv", index=False)
