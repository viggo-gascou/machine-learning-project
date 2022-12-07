import numpy as np
import matplotlib.pyplot as plt

from mlproject.helpers import data_loader, accuracy_score
from mlproject.neural_net import DenseLayer, NeuralNetworkClassifier

X_train, X_test, y_train, y_test = data_loader(raw=True, scaled=False)
X_train_SS, X_test_SS, y_train_SS, y_test_SS = data_loader(raw=False, scaled=True)
#X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = data_loader(raw=False, scaled=False, pca=True)

NN = NeuralNetworkClassifier(loss='cross_entropy')
NN.add(DenseLayer(784,128,"leaky_relu"))
NN.add(DenseLayer(128,32,"leaky_relu"))
NN.add(DenseLayer(32,16,"leaky_relu"))
NN.add(DenseLayer(16,5,"softmax"))

NN.fit(X_train_SS, y_train_SS, batches=5, epochs=150, lr=0.00001)

""" MAYBE DIFFERENT WEIGHT INITIALIZATION FOR SOFTMAX??"""



#print("LOSS HISTORY:")
#print(NN.loss_history)
#print(min(NN.loss_history),"epoch", NN.loss_history.index(min(NN.loss_history))+1)
#print("ACCURACY HISTORY:")
#print(NN.accuracy_history)

preds = NN.predict(X_train_SS)
print("ACCURACY ON TRAIN: ",accuracy_score(y_train_SS, preds))
np.savetxt("results/loss_hist.txt", NN.loss_history)
np.savetxt("results/acc_hist.txt", NN.accuracy_history)





n_hist = np.arange(len(NN.loss_history))
plt.subplots(figsize=(15,15))
plt.plot(n_hist, NN.loss_history)
plt.title("Loss history")
plt.savefig("results/loss_hist.png")

n_acc = np.arange(len(NN.accuracy_history))
plt.subplots(figsize=(15,15))
plt.plot(n_acc, NN.accuracy_history)
plt.title("Accuracy history")
plt.savefig("results/acc_hist.png")

