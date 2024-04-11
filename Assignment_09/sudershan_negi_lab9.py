import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

np.random.seed(0)

dataset = pd.read_excel("factoryReports.xlsx")
english_stop_words = set(stopwords.words('english'))

def preprocess_text(input_text):
    text_tokens = word_tokenize(input_text.lower())
    text_tokens = [token for token in text_tokens if token not in english_stop_words]
    processed_text =' '.join(text_tokens)
    return processed_text

dataset['Description'] = dataset['Description'].apply(preprocess_text)
dataset['Description'] = dataset['Description'].str.lower().str.split()
dataset['Category'] = dataset['Category'].str.lower().str.split()

vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(dataset['Description'].apply(' '.join))
labels = dataset['Category'].apply(' '.join)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(features_train, labels_train)

predicted_labels = classifier.predict(features_test)

accuracy = accuracy_score(labels_test, predicted_labels)
print("Accuracy is: ", accuracy)

unique_labels=np.unique(labels_test)

confusion_mat = confusion_matrix(labels_test, predicted_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()