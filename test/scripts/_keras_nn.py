import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlproject.helpers import data_loader, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
sns.set_style("darkgrid")

from keras.models import Sequential
from keras.layers import Dense, InputLayer, LeakyReLU
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD

def train_keras_nn():

    X_train_SS, X_test_SS, y_train_SS, y_test_SS = data_loader(raw=False, scaled=True)
    classes = (["T-shirt/top", "Trouser", "Pullover", "Dress", "Shirt"])


    # populate label-intcode dictionaries
    unique_classes = np.unique(y_train_SS)
    k = len(unique_classes)
    label = {k: unique_classes[k] for k in range(k)}
    intcode = {unique_classes[k]: k for k in range(k)}

    one_hot = OneHotEncoder(categories=[unique_classes])
    y_train_hot = one_hot.fit_transform(y_train_SS).toarray()
    y_test_hot = one_hot.fit_transform(y_test_SS).toarray()


    model = Sequential()
    model.add(InputLayer(input_shape=(784,)))
    model.add(Dense(32, activation=LeakyReLU(alpha=0.01), kernel_initializer='HeNormal'))
    model.add(Dense(16, input_dim=32, activation=LeakyReLU(alpha=0.01), kernel_initializer='HeNormal'))
    model.add(Dense(5, input_dim=16, activation='softmax', name='output', kernel_initializer='GlorotNormal'))


    model.compile(loss='categorical_crossentropy',
                    optimizer= SGD(0.001), 
                    metrics=['accuracy'])
    print("\n")
    model.summary()

    print("\nTRAINING\n")
    history = model.fit(X_train_SS, y_train_hot,
                batch_size=100,
                epochs=75,
                verbose=2
    ) 

    print("\nPREDICTING\n")
    y_pred = model.predict(X_test_SS)
    y_pred_labels = np.argmax(y_pred, axis=1)


    y_train_pred = model.predict(X_train_SS)
    y_pred_train_labels = np.argmax(y_train_pred, axis=1)
    print("\nFinal training accuracy: ",accuracy_score(y_train_SS, y_pred_train_labels),"\n")

    answer = input(f'Do you want to create and save the classification report? (y/n) ')  
    if answer == 'y' or answer == 'Y':
        test_report = classification_report(y_test_SS, y_pred_labels, target_names=classes)
        train_report = classification_report(y_train_SS, y_pred_train_labels, target_names=classes)
        print("\nClassification report for training data")
        print(train_report)
        print("Classification report for test data")
        print(test_report)
        test_report = pd.DataFrame(classification_report(y_test_SS, y_pred_labels, output_dict=True, target_names=classes)).transpose()
        train_report = pd.DataFrame(classification_report(y_train_SS, y_pred_train_labels, output_dict=True, target_names=classes)).transpose()
        test_report.to_csv("results/keras_FFNN/test_classification_report.csv")
        train_report.to_csv("results/keras_FFNN/train_classification_report.csv")

    answer = input(f'Do you want to create and save a plot of training accuracy and loss history? (y/n) ')  
    if answer == 'y' or answer == 'Y':
        sns.set_style("darkgrid")
        hist = history.history
        acc_hist, loss_hist = hist['accuracy'], hist['loss']
        epochs = np.arange(len(loss_hist))+1
        _, ax = plt.subplots(figsize=(8,8))
        ax.plot(epochs, acc_hist)
        ax.plot(epochs, loss_hist, c = 'r')
        plt.title("Keras CNN Training Accuracy and Loss History")
        plt.xlabel("Epoch")
        plt.savefig("results/keras_FFNN/training_hist.png")
        plt.show()

    answer = input(f'Do you want to create and save a confusion matrix? (y/n) ')  
    if answer == 'y' or answer == 'Y':  
        sns.set_style('white')
        cm = confusion_matrix(y_test_SS, y_pred_labels)
        disp = ConfusionMatrixDisplay(cm, display_labels = classes)
        _, ax = plt.subplots(figsize=(8,8))
        disp.plot(ax=ax)
        plt.title("Keras CNN Confusion Matrix")
        plt.savefig("results/keras_FFNN/confusion_matrix.png")
        plt.show()


if __name__ == '__main__':
    train_keras_nn()