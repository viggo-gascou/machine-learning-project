import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlproject.helpers import data_loader, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
sns.set_style("darkgrid")

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD

def train_keras_cnn():

    X_train_SS, X_test_SS, y_train_SS, y_test_SS = data_loader(raw=False, scaled=True)
    classes = (["T-shirt/top", "Trouser", "Pullover", "Dress", "Shirt"])


    #Reshape the input 
    X_train_CNN = X_train_SS.reshape(-1, 28, 28, 1)
    X_test_CNN = X_test_SS.reshape(-1, 28, 28, 1)

    # populate label-intcode dictionaries
    unique_classes = np.unique(y_train_SS)
    k = len(unique_classes)
    label = {k: unique_classes[k] for k in range(k)}
    intcode = {unique_classes[k]: k for k in range(k)}

    one_hot = OneHotEncoder(categories=[unique_classes])
    y_train_hot = one_hot.fit_transform(y_train_SS).toarray()
    y_test_hot = one_hot.fit_transform(y_test_SS).toarray()


    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(strides=2))
    model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(strides=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(5, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                    optimizer= Adam(0.001), 
                    metrics=['accuracy'])
    print("\n")
    model.summary()

    print("\nTRAINING\n")
    early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=5, restore_best_weights=True)
    history = model.fit(X_train_CNN, y_train_hot,
                batch_size=100,
                epochs=25,
                validation_split=0.2,
                verbose=2,
                callbacks=[early_stop]
    ) 

    print("\nPREDICTING\n")
    y_pred = model.predict(X_test_CNN)
    y_pred_labels = np.argmax(y_pred, axis=1)


    y_train_pred = model.predict(X_train_CNN)
    y_pred_train_labels = np.argmax(y_train_pred, axis=1)
    print("\nFinal training accuracy: ",accuracy_score(y_train_SS, y_pred_train_labels),"\n")

    answer = input(f'Do you want to create and save the classification report? (y/n) ')  
    if answer == 'y' or answer == 'Y':  
        report = classification_report(y_test_SS, y_pred_labels)
        print(report)
        report = pd.DataFrame(classification_report(y_test_SS, y_pred_labels, output_dict=True)).transpose()
        report.to_csv("results/keras_CNN_classification_report.csv")

    answer = input(f'Do you want to create and save a plot of training accuracy and loss history? (y/n) ')  
    if answer == 'y' or answer == 'Y':
        hist = history.history
        acc_hist, loss_hist = hist['accuracy'], hist['loss']
        epochs = np.arange(len(loss_hist))+1
        plt.plot(epochs, acc_hist)
        plt.plot(epochs, loss_hist, c = 'r')
        plt.title("Keras CNN Training Accuracy and Loss History")
        plt.xlabel("Epoch")
        plt.savefig("results/keras_CNN_training_hist.png")
        plt.show()

    answer = input(f'Do you want to create and save a confusion matrix? (y/n) ')  
    if answer == 'y' or answer == 'Y':  
        sns.set_style('white')
        cm = confusion_matrix(y_test_SS, y_pred_labels)
        disp = ConfusionMatrixDisplay(cm, display_labels = classes)
        disp.plot()
        plt.title("Keras CNN Confusion Matrix")
        plt.savefig("results/keras_CNN_confusion_matrix.png")
        plt.show()


if __name__ == '__main__':
    train_keras_cnn()