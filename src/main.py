
import numpy as np

from scripts.helpers import data_loader
from scripts.models.neural_net import DenseLayer
from scripts.models import NeuralNetworkClassifier

from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test = data_loader()

NN = NeuralNetworkClassifier(loss='cross_entropy')
NN.add(DenseLayer(784,5,"softmax"))
NN.fit(X_train, y_train, batches=1000, epochs=100)



preds = NN.predict(X_test)

print(accuracy_score(y_test, preds))

#print(DenseLayer(784,5,"softmax").activation_function())
