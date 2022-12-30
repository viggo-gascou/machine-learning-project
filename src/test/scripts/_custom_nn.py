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

    NN.fit(X_train_SS, y_train_SS, batches=100, epochs=75, lr=0.001)
    print("\nFinal training accuracy: ",accuracy_score(y_train_SS, NN.predict(X_train_SS)),"\n")


    y_preds = NN.predict(X_test_SS)

    answer = input(f'Do you want to create and save the classification report? (y/n) ')  
    if answer == 'y' or answer == 'Y':  
        report = classification_report(y_test_SS, y_preds)
        print(report)
        report = pd.DataFrame(classification_report(y_test_SS, y_preds, output_dict=True)).transpose()
        report.to_csv("results/custom_NN_classification_report.csv")

    answer = input(f'Do you want to create and save a plot of training accuracy and loss history? (y/n) ')  
    if answer == 'y' or answer == 'Y':  
        acc_hist = NN.accuracy_history
        loss_hist = NN.loss_history
        epochs = np.arange(len(loss_hist))+1
        plt.plot(epochs, acc_hist)
        plt.plot(epochs, loss_hist, c = 'r')
        plt.title("Custom NN Training Accuracy and Loss History")
        plt.xlabel("Epoch")
        plt.savefig("results/custom_NN_training_hist.png")
        plt.show()

    answer = input(f'Do you want to create and save a confusion matrix? (y/n) ')  
    if answer == 'y' or answer == 'Y':  
        sns.set_style('white')
        cm = confusion_matrix(y_test_SS, y_preds)
        disp = ConfusionMatrixDisplay(cm, display_labels = classes)
        disp.plot()
        plt.title("Custom NN Confusion Matrix")
        plt.savefig("results/custom_NN_confusion_matrix.png")
        plt.show()


if __name__ == '__main__':
    train_custom_nn()