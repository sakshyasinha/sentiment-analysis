# %% Import libraries
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# %% Load dataset
dataset = pd.read_csv('Tweets.csv')

# Drop rows with missing sentiment labels
dataset.dropna(subset=['airline_sentiment'], inplace=True)

# %% Text preprocessing
nltk.download('stopwords')
corpus = []

for i in range(0, min(1000, len(dataset))):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'].iloc[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    if 'not' in all_stopwords:
        all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# %% Feature extraction
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()

# Target variable: airline_sentiment
y_raw = dataset.iloc[:len(corpus)]['airline_sentiment'].values

# Encode sentiment labels to integers
le = LabelEncoder()
y = le.fit_transform(y_raw)

# %% Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# %% Train classifier
classifier = MultinomialNB()
classifier.fit(x_train, y_train)

# %% Predict and evaluate
y_pred = classifier.predict(x_test)

# Decode labels
decoded_pred = le.inverse_transform(y_pred)
decoded_test = le.inverse_transform(y_test)

# Show paired predictions vs true labels
paired = np.concatenate((decoded_pred.reshape(-1, 1), decoded_test.reshape(-1, 1)), axis=1)
print("Predicted vs Actual Sentiments:\n", paired)

# Show metrics
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
