from mlproject.helpers import data_loader, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np

X_train, X_test, y_train, y_test = data_loader(raw=False, scaled=False, pca=True)


DT = DecisionTreeClassifier(criterion='gini', min_samples_leaf=2, max_depth=2)

DT.fit(X_train, y_train)

y_preds = DT.predict(X_test)

print(accuracy_score(y_test, y_preds))

classification_report(y_test, y_preds)