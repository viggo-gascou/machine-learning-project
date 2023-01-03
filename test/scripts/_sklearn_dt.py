from sklearn.tree import DecisionTreeClassifier
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlproject.helpers import data_loader, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def train_sklearn_dt():
    print()
    print("-"*32)
    print("SCIKIT-LEARN DECISION TREE CLASSIFIER")
    print("-"*32)

    X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = data_loader(raw=False, scaled=False, pca=True)
    X_train_PCA, X_test_PCA = X_train_PCA[:,:62], X_test_PCA[:,:62]
    classes = (["T-shirt/top", "Trouser", "Pullover", "Dress", "Shirt"])


    DT = DecisionTreeClassifier(criterion='gini', max_depth=15, min_samples_leaf=21)


    DT.fit(X_train_PCA, y_train_PCA)
    
    y_preds = DT.predict(X_test_PCA)
    y_train_preds = DT.predict(X_train_PCA)
    print("\nFinal training accuracy: ",accuracy_score(y_train_PCA, y_train_preds),"\n")


    answer = input(f'Do you want to create and save the classification report? (y/n) ')  
    if answer == 'y' or answer == 'Y':  
        test_report = classification_report(y_test_PCA, y_preds, target_names=classes)
        train_report = classification_report(y_train_PCA, y_train_preds, target_names=classes)
        print("Classification report for training data")
        print(train_report)
        print("Classification report for test data")
        print(test_report)
        test_report = pd.DataFrame(classification_report(y_test_PCA, y_preds, output_dict=True, target_names=classes)).transpose()
        train_report = pd.DataFrame(classification_report(y_train_PCA, y_train_preds, output_dict=True, target_names=classes)).transpose()
        test_report.to_csv("results/sklearn_DT/test_classification_report.csv")
        train_report.to_csv("results/sklearn_DT/train_classification_report.csv")
        
    answer = input(f'Do you want to create and save a confusion matrix? (y/n) ')  
    if answer == 'y' or answer == 'Y':  
        cm = confusion_matrix(y_test_PCA, y_preds)
        disp = ConfusionMatrixDisplay(cm, display_labels = classes)
        _, ax = plt.subplots(figsize=(8,8))
        disp.plot(ax=ax)
        plt.title("Custom DT Confusion Matrix")
        plt.savefig("results/sklearn_DT/confusion_matrix.png")
        plt.show()


if __name__ == '__main__':
    train_sklearn_dt()