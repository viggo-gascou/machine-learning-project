import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlproject.helpers import data_loader, accuracy_score
from mlproject.decision_tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
sns.set_style("darkgrid")

def train_custom_dt():
    print()
    print("-"*32)
    print("CUSTOM DECISION TREE CLASSIFIER")
    print("-"*32)

    X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = data_loader(raw=False, scaled=False, pca=True)
    X_train_PCA, X_test_PCA = X_train_PCA[:,:62], X_test_PCA[:,:62]
    classes = (["T-shirt/top", "Trouser", "Pullover", "Dress", "Shirt"])


    DT = DecisionTreeClassifier(criterion='gini', max_depth=15, min_samples_in_leaf=21)


    DT.fit(X_train_PCA, y_train_PCA)
    print("\nFinal training accuracy: ",accuracy_score(y_train_PCA, DT.predict(X_train_PCA)),"\n")
    
    y_preds = DT.predict(X_test_PCA)


    answer = input(f'Do you want to create and save the classification report? (y/n) ')  
    if answer == 'y' or answer == 'Y':  
        report = classification_report(y_test_PCA, y_preds)
        print(report)
        report = pd.DataFrame(classification_report(y_test_PCA, y_preds, output_dict=True)).transpose()
        report.to_csv("results/custom_DT_classification_report.csv")
        
    answer = input(f'Do you want to create and save a confusion matrix? (y/n) ')  
    if answer == 'y' or answer == 'Y':  
        cm = confusion_matrix(y_test_PCA, y_preds)
        disp = ConfusionMatrixDisplay(cm, display_labels = classes)
        disp.plot()
        plt.title("Custom DT Confusion Matrix")
        plt.savefig("results/DT_confusion_matrix.png")
        plt.show()


if __name__ == '__main__':
    train_custom_dt()