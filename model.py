import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

data = pd.read_csv('D:/PycharmProjectsDD/logreg_api/data/sportset.csv', sep=";")
data['sport'] = np.where(data['sport'] == 'basket', 1, 0)
y = data['sport']
X = data[['height', 'weight']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

y_preds = clf.predict(X_test)

print(confusion_matrix(y_test, y_preds))

precision = precision_score(y_test, y_preds)
recall = recall_score(y_test, y_preds)

print('precision is {:.2f}'.format(precision))
print('recall is {:.2f}'.format(recall))


