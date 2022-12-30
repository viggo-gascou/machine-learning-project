

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder

from mlproject.helpers import data_loader, accuracy_score

# install tensorflow-macos version 2.9.0 and tensorflow-metal version 0.5.0:
    # python -m pip install tensorflow-macos==2.9.0
    # python -m pip install tensorflow-metal==0.5.0
# alongside (preferably before) with tensorflow dependencies: 
    # conda install -c apple tensorflow-deps
# latest probably working version can be found at https://developer.apple.com/metal/tensorflow-plugin/ (near the bottom)

print("\nLOADING TENSORFLOW")
import tensorflow as tf 

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.callbacks import EarlyStopping

from keras.optimizers import Adam, SGD


print("\nLOADING DATA")
X_train, X_test, y_train, y_test = data_loader(raw=False, scaled=True)

#Reshape the input 
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# populate label-intcode dictionaries
unique_classes = np.unique(y_train)
k = len(unique_classes)
label = {k: unique_classes[k] for k in range(k)}
intcode = {unique_classes[k]: k for k in range(k)}

one_hot = OneHotEncoder(categories=[unique_classes])
y_train_hot = one_hot.fit_transform(y_train).toarray()
y_test_hot = one_hot.fit_transform(y_test).toarray()


print("\nINITIALIZING MODEL\n")
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
history = model.fit(X_train, y_train_hot,
            batch_size=100,
            epochs=25,
            validation_split=0.2,
            verbose=1,
            callbacks=[early_stop]
) 

print("\nPREDICTING\n")
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y2label = ["T-shirt/top",
           "Trouser",
           "Pullover",
           "Dress",
           "Shirt"]
print(accuracy_score(y_test, y_pred_labels))
y_train_pred = model.predict(X_train)
y_pred_train_labels = np.argmax(y_train_pred, axis=1)
print(accuracy_score(y_train, y_pred_train_labels))
CM = confusion_matrix(y_test, y_pred_labels)
disp = ConfusionMatrixDisplay(CM, display_labels=y2label)
disp.plot()
plt.savefig("CNN_confusion_matrix.png")

acc_hist = history.history['accuracy']
loss_hist = history.history['loss']
sns.set_style("darkgrid")

plt.subplots(figsize=(10,6))
plt.title("Training Accuracy and Loss History for LeNet-5")
plt.xticks(range(1,len(loss_hist)+1))
plt.plot(range(1,len(acc_hist)+1), acc_hist, c="r", label='Training Accuracy')
plt.plot(range(1,len(loss_hist)+1), loss_hist, label='Trainig Loss')
plt.xlabel("Epochs")
plt.legend()
plt.savefig("CNN_train_hist.png")
print(classification_report(y_test, y_pred_labels))