import numpy as np
from mlproject.helpers import data_loader, accuracy_score
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD

from sklearn.metrics import classification_report


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
model.add(Dense(128, input_dim=X_train_SS.shape[1], activation='relu', kernel_initializer='HeNormal'))
model.add(Dense(32, input_dim=128, activation='relu', kernel_initializer='HeNormal'))
model.add(Dense(16, input_dim=32, activation='relu', kernel_initializer='HeNormal'))
model.add(Dense(5, input_dim=16, activation='softmax', name='output', kernel_initializer='GlorotNormal'))


model.compile(loss='categorical_crossentropy',
                optimizer= SGD(0.0001), 
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
print("\nFinal training accuracy: ", accuracy_score(y_train_SS, y_pred_train_labels),"\n")
print("Test acc:",accuracy_score(y_test_SS, y_pred_labels))

print()