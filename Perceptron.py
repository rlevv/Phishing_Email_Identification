import numpy as np
import random
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# abstract perceptron class
class Perceptron(ABC):
    """
    Abstract class for different perceptron training methods.

    This class has a number of subclasses intended to override the train method.

    Attributes:
        W: a numpy array representing the weight vector
        bias: the bias
        epochs: the number of training epochs
        lr: the learning rate
        accuracies: a list of accuracies for each epoch
    """
    def __init__(self, num_features, lr, epochs):
        """
        Constructor for Perceptron class.

        Accepts basic information for the perceptron and initializes the instance variables.

        Args:
            num_features: the number of data features
            lr: the learning rate
            epochs: the number of training epochs
        """
        self.W = np.random.uniform(-.01,.01,num_features) 
        self.bias = random.uniform(-.01,.01)
        self.epochs = epochs
        self.lr = lr

        self.accuracies = {}

    def __repr__(self):
        """
        Print function for Perceptron.

        Returns:
            A tuple where the first entry is the Perceptron weight and 
            the second is the bias.
        """
        return (self.W, self.bias)
    
    def test(self, test_data, test_labels):
        """
        Tests the perceptron on a given data and label set.

        Args:
            test_data: the number of data features
            lr: the learning rate
            epochs: the number of training epochs

        Returns:
            A tuple where the first entry is the Perceptron weight and 
            the second is the bias.
        """
        num_examples = test_data.shape[0]
        misclassifications = 0

        for i in range(num_examples):
            example = test_data[i]
            actual_label = test_labels[i]
            predicted_label = self.W.T.dot(example) + self.bias
            
            if (actual_label * predicted_label) < 0:
                self.W = self.W + (self.lr * actual_label * example)
                self.bias = self.bias + (self.lr * actual_label)
                    
                misclassifications += 1
        
        accuracy = (num_examples - misclassifications)/num_examples
        print("accuracy on test set is", accuracy, ".")
        return accuracy
    
    @abstractmethod
    def train(self):
        """
        Abstract method to be overridden in subclasses
        """
        pass
    
    def graph(self):
        """
        Graphs the accuracy across the epochs
        """
        x = list(self.accuracies.keys())
        y = list(self.accuracies.values())
        
        plt.plot(x,y)
        plt.xlabel("Epoch")
        plt.ylabel("Dev Set Accuracy")
        plt.title("Dev Set Accuracy Across Epochs")
        plt.show()

class SimplePerceptron(Perceptron):
    def __init__(self, num_features, lr=.01, epochs=10):
        super(SimplePerceptron, self).__init__(num_features, lr, epochs)
        
    def train(self, train_data, train_labels, dev_data, dev_labels):
        num_examples = train_data.shape[0]
        
        best_accuracy = [0, [], 0]
        
        for epoch in range(self.epochs):
            misclassifications = 0
            
            for i in range(num_examples):
                example = train_data[i]
                actual_label = train_labels[i]
                predicted_label = self.W.T.dot(example) + self.bias
                
                if (actual_label * predicted_label) < 0:
                    self.W = self.W + (self.lr * actual_label * example)
                    self.bias = self.bias + (self.lr * actual_label)
                    
                    misclassifications += 1
            
            if misclassifications == 0:
                print("epoch", epoch, "completed with 100% accuracy.")
                return (self.W, self.bias)
            else:
                accuracy = (num_examples - misclassifications)/num_examples
                self.accuracies[epoch] = accuracy
                if accuracy > best_accuracy[0]:
                    best_accuracy[0] = accuracy
                    best_accuracy[1] = self.W
                    best_accuracy[2] = self.bias
                print("epoch", epoch, "complete with accuracy", accuracy, ".")
        self.W = best_accuracy[1]
        self.bias = best_accuracy[2]
        return (self.W, self.bias)

x = SimplePerceptron(100, .001, 100)