import numpy as np
import matplotlib.pyplot as plt

from mlproject.helpers import data_loader, accuracy_score
from mlproject.neural_net import DenseLayer, NeuralNetworkClassifier
from mlproject.decision_tree import DecisionTreeClassifier
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


#X_train, X_test, y_train, y_test = data_loader(raw=True, scaled=False)
X_train_SS, X_test_SS, y_train_SS, y_test_SS = data_loader(raw=False, scaled=True)
#X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = data_loader(raw=False, scaled=False, pca=True)

#X_train_SS, X_val_SS, y_train_SS, y_val_SS = train_test_split(X_train_SS, y_train_SS, test_size=0.2, random_state=42, stratify=y_train_SS)


NN = NeuralNetworkClassifier(loss='cross_entropy')
NN.add(DenseLayer(784,32,"leaky_relu"))
NN.add(DenseLayer(32,16,"leaky_relu"))
NN.add(DenseLayer(16,5,"softmax"))

NN.fit(X_train_SS, y_train_SS, batches=1, epochs=75, lr=0.001)

# test_preds = NN.predict(X_test_SS)

# CR = classification_report(y_test_SS, test_preds, output_dict=True)
# print(classification_report(y_test_SS, test_preds))
# df = pd.DataFrame(CR).transpose()
# print(df.style.to_latex())

# print(NN.accuracy_history)

# with open("acc_hist.pkl", "wb") as f:
#     pickle.dump(NN.accuracy_history, f)

# with open("loss_hist.pkl", "wb") as f:
#     pickle.dump(NN.loss_history, f)

#Tree = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_in_leaf=2)
#Tree.fit(X_train_PCA, y_train_PCA)
#preds = Tree.predict_proba(X_train_PCA)
#print(preds)

# print("LOSS HISTORY:")
# print(NN.loss_history)
# print(min(NN.loss_history),"epoch", NN.loss_history.index(min(NN.loss_history))+1)
# print("ACCURACY HISTORY:")
# print(NN.accuracy_history)
# train_preds = NN.predict(X_train_SS)
# val_preds = NN.predict(X_val_SS)
# 

# print(preds)
# print("ACCURACY ON TRAIN: ",accuracy_score(y_train_SS, train_preds))
# print("ACCURACY ON VALIDATION: ",accuracy_score(y_val_SS, val_preds))
# print("ACCURACY ON TEST: ",accuracy_score(y_test_SS, test_preds))
#np.savetxt("results/loss_hist.txt", NN.loss_history)
#np.savetxt("results/acc_hist.txt", NN.accuracy_history)





# n_hist = np.arange(len(NN.loss_history))
# plt.subplots(figsize=(15,15))
# plt.plot(n_hist, NN.loss_history)
# plt.title("Loss history")
# plt.savefig("results/loss_hist.png")

# n_acc = np.arange(len(NN.accuracy_history))
# plt.subplots(figsize=(15,15))
# plt.plot(n_acc, NN.accuracy_history)
# plt.title("Accuracy history")
# plt.savefig("results/acc_hist.png")

