import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline


col = ["text", "label"]
start_time = time.time()
df = pd.read_csv("Dataset Path", names=['text', 'label'])

df = shuffle(df, random_state=22)
X = df['text']
y = df['label']

t = TfidfVectorizer(use_idf=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
X_train = t.fit_transform(X_train.values.astype('U'))

lr = LogisticRegression(random_state=0, max_iter=300)
lr.fit(X_train, y_train)

X_test = t.transform(X_test.values.astype('U'))
y_pred = lr.predict(X_test)


X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.20, random_state=0)
pipe = Pipeline([('Vectorizer', t), ('Model', lr)])
pipe.fit(X_train1.values.astype('U'), y_train1)
y_pred1 = pipe.predict(X_test1.values.astype('U'))

joblib.dump(pipe, "Ml_Pipeline")

