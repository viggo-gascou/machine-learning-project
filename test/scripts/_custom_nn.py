import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlproject.helpers import data_loader, accuracy_score
from mlproject.neural_net import DenseLayer, NeuralNetworkClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
sns.set_style("darkgrid")

def train_custom_nn():
    print()
    print("-"*46)
    print("CUSTOM FEED FORWARD NEURAL NETWORK CLASSIFIER")
    print("-"*46)

    X_train_SS, X_test_SS, y_train_SS, y_test_SS = data_loader(raw=False, scaled=True)
    classes = (["T-shirt/top", "Trouser", "Pullover", "Dress", "Shirt"])


    NN = NeuralNetworkClassifier(loss='cross_entropy')
    NN.add(DenseLayer(784,32,"leaky_relu"))
    NN.add(DenseLayer(32,16,"leaky_relu"))
    NN.add(DenseLayer(16,5,"softmax"))

    NN.fit(X_train_SS, y_train_SS, batches=100, epochs=75, lr=0.0001)

    y_train_preds = NN.predict(X_train_SS)
    y_preds = NN.predict(X_test_SS)
    print("\nFinal training accuracy: ",accuracy_score(y_train_SS, y_train_SS),"\n")

    answer = input(f'Do you want to create and save the classification report? (y/n) ')  
    if answer == 'y' or answer == 'Y':  
        test_report = classification_report(y_test_SS, y_preds, target_names=classes)
        train_report = classification_report(y_train_SS, y_train_preds, target_names=classes)
        print("\nClassification report for training data")
        print(train_report)
        print("Classification report for test data")
        print(test_report)
        test_report = pd.DataFrame(classification_report(y_test_SS, y_preds, output_dict=True, target_names=classes)).transpose()
        train_report = pd.DataFrame(classification_report(y_train_SS, y_train_preds, output_dict=True, target_names=classes)).transpose()
        test_report.to_csv("results/custom_NN/test_classification_report.csv")
        train_report.to_csv("results/custom_NN/train_classification_report.csv")

    answer = input(f'Do you want to create and save a plot of training accuracy and loss history? (y/n) ')  
    if answer == 'y' or answer == 'Y':  
        acc_hist = NN.accuracy_history
        loss_hist = NN.loss_history
        epochs = np.arange(len(loss_hist))+1
        _, ax = plt.subplots(figsize=(8,8))
        ax.plot(epochs, acc_hist)
        ax.plot(epochs, loss_hist, c = 'r')
        plt.title("Custom NN Training Accuracy and Loss History")
        plt.xlabel("Epoch")
        plt.savefig("results/custom_NN/training_hist.png")
        plt.show()

    answer = input(f'Do you want to create and save a confusion matrix? (y/n) ')  
    if answer == 'y' or answer == 'Y':  
        sns.set_style('white')
        cm = confusion_matrix(y_test_SS, y_preds)
        disp = ConfusionMatrixDisplay(cm, display_labels = classes)
        _, ax = plt.subplots(figsize=(8,8))
        disp.plot(ax=ax)
        plt.title("Custom NN Confusion Matrix")
        plt.savefig("results/custom_NN/confusion_matrix.png")
        plt.show()


if __name__ == '__main__':
    train_custom_nn()