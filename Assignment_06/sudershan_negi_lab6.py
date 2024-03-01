import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("tblTrain.csv")
test = pd.read_csv("tblTest.csv")

drama_movies = train[train['Genre'] == 'Drama']
action_movies = train[train['Genre'] == 'Action']
plt.subplot(2, 1, 1)
plt.scatter(action_movies['Duration'], action_movies['Rating'], label='Action', c='blue', marker='')
plt.scatter(drama_movies['Duration'], drama_movies['Rating'], label='Drama', c='red', marker='')
for index, movie in train.iterrows():
    color = 'blue' if movie['Genre'] == 'Action' else 'red'
    plt.annotate(movie['Name'], (movie['Duration'], movie['Rating']), textcoords="offset points", xytext=(0, 5), ha='center', color=color, fontsize=4)
plt.xlabel('Duration')
plt.ylabel('Rating')
plt.title('Train - Movie Ratings vs. Duration')

drama_movies = test[test['Genre'] == 'Drama']
action_movies = test[test['Genre'] == 'Action']
plt.subplot(2, 1, 2)
plt.scatter(action_movies['Duration'], action_movies['Rating'], label='Action', c='blue', marker='')
plt.scatter(drama_movies['Duration'], drama_movies['Rating'], label='Drama', c='red', marker='')
for index, movie in test.iterrows():
    color = 'blue' if movie['Genre'] == 'Action' else 'red'
    plt.annotate(movie['Name'], (movie['Duration'], movie['Rating']), textcoords="offset points", xytext=(0, 5), ha='center', color=color, fontsize=4)
plt.xlabel('Duration')
plt.ylabel('Rating')
plt.title('Test - Movie Ratings vs. Duration')

plt.subplots_adjust(wspace=0.5, hspace=0.5)
handles = [plt.Line2D([], [], marker='s', color='blue', markersize=10, linestyle='', label='Action'),
           plt.Line2D([], [], marker='s', color='red', markersize=10, linestyle='', label='Drama')]
plt.legend(handles=handles)
plt.xticks(np.arange(40, 260, 20))
plt.grid(True)
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

knn_neighbors = list(range(1, 500, 2))

accuracies=[]
for k in knn_neighbors:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(train[['Duration', 'Rating']], train['Genre'])
    y_pred = knn.predict(test[['Duration', 'Rating']])
    mat=confusion_matrix(test['Genre'], y_pred)
    accuracy= (mat[0, 0] + mat[1, 1]) / np.sum(mat)
    accuracies.append(accuracy)

plt.plot(knn_neighbors, accuracies)
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Neighbors (K)')
plt.yticks(np.arange(0.5, 1.1, 0.1))
plt.grid(True)
plt.show()

index=accuracies.index(max(accuracies))
k=knn_neighbors[index]
print(f'The value of K is {k} and the corresponding max accuracy is {max(accuracies)}')

knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(train[['Duration', 'Rating']], train['Genre'])
y_pred = knn.predict(test[['Duration', 'Rating']])
mat=confusion_matrix(test['Genre'], y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=mat, display_labels=['Action', 'Drama'])
disp.plot()
plt.title('Confusion Matrix')
plt.show()

precision = precision_score(test['Genre'], y_pred, average=None)
recall = recall_score(test['Genre'], y_pred, average=None)
f_score = f1_score(test['Genre'], y_pred, average=None)
print("\nPrecision for each class:", precision)
print("Recall for each class:", recall)
print("F-score for each class:", f_score)

overall_precision = precision_score(test['Genre'], y_pred, average='weighted')
overall_recall = recall_score(test['Genre'], y_pred, average='weighted')
overall_f_score = f1_score(test['Genre'], y_pred, average='weighted')
print("\nOverall Precision:", overall_precision)
print("Overall Recall:", overall_recall)
print("Overall F-score:", overall_f_score)