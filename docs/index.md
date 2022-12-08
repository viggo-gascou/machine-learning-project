# Welcome to the documentation for our machine learning project!

## Implementations

### [`Neural Network Classifier`][mlproject.neural_net._neural_net.NeuralNetworkClassifier]
A neural network for use in classification problems.

* [`DenseLayer`][mlproject.neural_net._dense_layer.DenseLayer] - A fully connected layer with user defined size and activation function

* [`Loss Functions`][mlproject.neural_net._loss.cross_entropy_loss] - The loss function to use in the neural network
    * (currently only [`categorical cross entropy`][mlproject.neural_net._loss.cross_entropy_loss] is supported).
  
* [`Activation Functions`][mlproject.neural_net._activations.leaky_relu] - The activation functions to use in the fully connected layers
    * (currently [`stable_softmax`][mlproject.neural_net._activations.stable_softmax] and [`leaky_relu`][mlproject.neural_net._activations.leaky_relu] is supported).

### [`Decision Tree Classifier`][mlproject.decision_tree._decision_tree.DecisionTreeClassifier]
A decision tree for use in classification problems.

* [`Node`][mlproject.decision_tree._node.Node] - A tree node used by the decision tree classifier, is either leaf or not.

* [`Impurtiy Criterion`][mlproject.decision_tree._impurity.entropy_impurity] - The impurity function to use when decision whether to split nodes in the decision tree 
    * (currently only  [`gini_impurity`][mlproject.decision_tree._impurity.gini_impurity] and  [`entropy`][mlproject.decision_tree._impurity.entropy_impurity] are supported).
  