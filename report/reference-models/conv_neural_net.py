

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
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
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
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
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPool2D(strides=2))
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
history = model.fit(X_train, y_train_hot,
            batch_size=100,
            epochs=20,
            verbose=1,
) 

acc_hist = history.history['accuracy']
loss_hist = history.history['loss']
plt.plot(range(len(acc_hist)), acc_hist)
plt.plot(range(len(loss_hist)), loss_hist, c="r")
plt.show()

print("\nPREDICTING\n")
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
print(accuracy_score(y_test, y_pred_labels))