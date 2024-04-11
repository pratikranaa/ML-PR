# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Downloading nltk resources
nltk.download('punkt')
nltk.download('stopwords')

# Setting random seed
np.random.seed(0)

# Loading data and preprocessing
data = pd.read_excel('factoryReports.xlsx')

stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    words = word_tokenize(text)  # Tokenize
    words = [word.lower() for word in words]  # Convert to lowercase
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Applying preprocessing
data['Description'] = data['Description'].apply(preprocess_text)
data['Category'] = data['Category'].apply(lambda x: x.lower())

# TF-IDF Vectorization and splitting data
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['Description'])
y = data['Category']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(x_train, y_train)

# Making predictions
y_pred = rf_classifier.predict(x_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plotting confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(np.arange(len(data['Category'].unique())), data['Category'].unique(), rotation=45)
plt.yticks(np.arange(len(data['Category'].unique())), data['Category'].unique())
for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix[i])):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red')
plt.show()
