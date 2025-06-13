import pandas as pd
import numpy as np
 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

true_df['Label'] = 'REAL'
fake_df['Label'] = 'FAKE'

df = pd.concat([true_df, fake_df], axis=0)
df = df.sample(frac=1).reset_index(drop=True)

df.head()

x = df['text']
y = df['Label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english' , max_df=0.7)

x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x_train_tfidf, y_train)

y_pred = model.predict(x_test_tfidf)

score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(score*100, 2)}%")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

def predict_news(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return prediction[0]

# Try your own article or headline
sample = "You won’t believe what this man did to cure cancer overnight!"
print("Prediction:", predict_news(sample))