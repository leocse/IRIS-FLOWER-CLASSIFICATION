import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt


iris = load_iris()


df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target


print("Dataset Info:")
print(df.info())
print("\nFirst few rows of the dataset:")
print(df.head())


X = df.drop('species', axis=1)  
y = df['species'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


new_sample = np.array([[5.1, 3.5, 1.4, 0.2]]) 
predicted_species = model.predict(new_sample)
print(f"\nPredicted species for the new sample: {iris.target_names[predicted_species][0]}")
