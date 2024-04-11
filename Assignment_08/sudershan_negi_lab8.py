import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score

df=pd.read_excel("data 1.xlsx")
palette={1: 'blue', 2:'orange', 3: 'green', 4:'red'}

sns.scatterplot(data=df, x='Clump thickness', y='No of week', hue=df['Cancer stage'], palette=palette)
plt.grid(True)
plt.xlabel("Clump thickness of the cell (in millimetres)")
plt.ylabel("Number of weeks since the formation of the clump")
plt.title("Labeled cancer stage of the cells (Total instances: 200)")
plt.xticks(np.arange(0, 20))
plt.yticks(np.arange(0, 21, 2.5))
plt.legend(title='Cancer Stage', loc='upper right')
plt.show()

sns.scatterplot(data=df, x='Clump thickness', y='No of week', hue=df['Cancer stage'], palette=palette)
sns.scatterplot(data=df, x='Clump thickness_new', y='No of week_new', color='grey', s=3)
plt.scatter([], [], color='grey', label='Unlabeled Data', s=3)
plt.grid(True)
plt.xlabel("Clump thickness of the cell (in millimetres)")
plt.ylabel("Number of weeks since the formation of the clump")
plt.title("Labeled cancer stage of the cells (New instances added: 2000)")
plt.xticks(np.arange(0, 20))
plt.yticks(np.arange(0, 21, 2.5))
plt.legend(title='Cancer Stage', loc='upper right')
plt.show()

x=df[['Clump thickness', 'No of week']][:200]
y=df['Cancer stage'][:200]

knn=KNeighborsClassifier()
ssm=SelfTrainingClassifier(base_estimator=knn)
ssm.fit(x, y)

sns.scatterplot(data=df, x='Clump thickness', y='No of week', hue=df['Cancer stage'], palette=palette)

df[['Clump thickness', 'No of week']]=df[['Clump thickness_new', 'No of week_new']]

y_train=df[['Clump thickness', 'No of week']]
y_pred=ssm.predict(y_train)

sns.scatterplot(data=df, x='Clump thickness', y='No of week', hue=y_pred, palette=palette)
plt.grid(True)
plt.xlabel("Clump thickness of the cell (in millimetres)")
plt.ylabel("Number of weeks since the formation of the clump")
plt.title("Fitted labels for unlabelled Data by Semi-supervised Learning method - cell cancer stage")
plt.xticks(np.arange(0, 20))
plt.yticks(np.arange(0, 21, 2.5))
plt.legend().set_visible(False)
plt.show()

cm=confusion_matrix(df['True cancer stage'], y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print(f'Classification Report : {classification_report(df["True cancer stage"], y_pred)}')

report_dict = classification_report(df["True cancer stage"], y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict)
sns.heatmap(report_df, annot=True, cmap='viridis')
plt.title('Classification Report')
plt.show()

print(f'Accuracy Score: {accuracy_score(df["True cancer stage"], y_pred)}')

print(f'Balanced Accuracy Score: {balanced_accuracy_score(df["True cancer stage"], y_pred)}')

