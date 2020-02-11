import numpy as np
import random
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# abstract perceptron class
class Perceptron(ABC):
    """
    Abstract class for different perceptron training methods.

    This class has a number of subclasses intended to override the 
    train method.

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

        Accepts basic information for the perceptron and initializes 
        the instance variables.

        Args:
            num_features: the number of data features
            lr: the learning rate
            epochs: the number of training epochs
        """
        self.W = np.random.uniform(-.01,.01,num_features) 
        self.bias = random.uniform(-.01,.01)
        self.epochs = epochs
        self.lr = lr
        self.accuracies = [[x for x in range(epochs + 1)] 
                                for y in range(3)]

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
            test_data: the test data
            test_labels: the corresponding test label array

        Returns:
            The accuracy from running the perceptron on a given dataset
        """
        num_examples = test_data.shape[0]

        misclassifications = 0
        for index, row in test_data.iterrows():
            example = np.asarray(row)
            label = test_labels.iat[index,0]
            predicted_label = self.W.T.dot(example) + self.bias
            
            if (label * predicted_label) < 0:
                misclassifications += 1
        
        accuracy = ((num_examples - misclassifications) / num_examples)
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
        x = self.accuracies[0]
        y1 = self.accuracies[1]
        y2 = self.accuracies[2]

        plt.plot(x, y1, color='blue', label = "Training Set")
        plt.plot(x, y2, color='red', label = "Dev Set")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Across Epochs")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.show()

class SimplePerceptron(Perceptron):
    """
    A Perceptron subclass that implements a simple perceptron.

    This class inherits most methods from Perceptron abstract class.
    It overrides the train method.
    """
    def __init__(self, num_features, lr=.01, epochs=10):
        """
        Constructor for SimplePerceptron class.

        Accepts basic information for the simple perceptron and 
        instantiates according to the Perceptron abstract class. 

        Args:
            num_features: the number of data features
            lr: the learning rate which defaults to .01
            epochs: the number of training epochs which defaults 10
        """
        super(SimplePerceptron, self).__init__(num_features, 
                                               lr, epochs)
        
    def train(self, train_data, train_labels, dev_data, dev_labels):
        """
        Trains a simple perceptron on a given data and label set.

        Args:
            train_data: the train data
            train_labels: the corresponding trian label array

        Returns:
            A tuple where the first entry is the Perceptron weight and 
            the second is the bias.
        """
        initial_train_accuracy = self.test(train_data, train_labels)
        initial_dev_accuracy = self.test(dev_data, dev_labels)

        self.accuracies[1][0] = initial_train_accuracy
        self.accuracies[2][0] = initial_dev_accuracy

        num_examples = train_data.shape[0]
        
        for epoch in range(self.epochs):
            misclassifications = 0
            for index, row in train_data.iterrows():
                example = np.asarray(row)
                label = train_labels.iat[index,0]
                predicted_label = self.W.T.dot(example) + self.bias
                if (label * predicted_label) < 0:
                    self.W = self.W + (self.lr * label \
                                       * example)
                    self.bias = self.bias + (self.lr * label)
                    misclassifications += 1

            train_accuracy = ((num_examples - misclassifications) \
                             / num_examples)
            dev_accuracy = self.test(dev_data, dev_labels)
            
            self.accuracies[1][epoch+1] = train_accuracy
            self.accuracies[2][epoch+1] = dev_accuracy

        return (self.W, self.bias)

class DynamicPerceptron(Perceptron):
    """
    A Perceptron subclass that implements a dynamic perceptron.

    This class inherits most methods from Perceptron abstract class.
    It overrides the train method.
    """
    def __init__(self, num_features, lr=.01, epochs=10):
        """
        Constructor for DynamicPerceptron class.

        Accepts basic information for the simple perceptron and 
        instantiates according to the Perceptron abstract class. 

        Args:
            num_features: the number of data features
            lr: the learning rate which defaults to .01
            epochs: the number of training epochs which defaults 10
        """
        super(DynamicPerceptron, self).__init__(num_features, lr, 
                                                epochs)
        
    def train(self, train_data, train_labels):
        """
        Trains a dynamic perceptron on a given data and label set.

        Args:
            train_data: the train data
            train_labels: the corresponding trian label array

        Returns:
            A tuple where the first entry is the Perceptron weight and
            the second is the bias.
        """
        timestep = 0
        num_examples = train_data.shape[0]
        
        best_accuracy = [0, [], 0]
        
        for epoch in range(self.epochs):
            misclassifications = 0
            
            for i in range(num_examples):
                example = train_data[i]
                actual_label = train_labels[i]
                predicted_label = self.W.T.dot(example) + self.bias
                
                if (actual_label * predicted_label) < 0:
                    adjusted_lr = self.lr/(1+timestep)
                    self.W = self.W + (adjusted_lr * actual_label 
                                       * example)
                    self.bias = self.bias + (adjusted_lr 
                                             * actual_label)
                    
                    misclassifications += 1
            
            if misclassifications == 0:
                print("epoch", epoch, "completed with perfect \
                      accuracy.")
                return (self.W, self.bias)
            else:
                timestep += 1 
                accuracy = (num_examples - misclassifications) \
                            / num_examples
                self.accuracies[epoch] = accuracy
                if accuracy > best_accuracy[0]:
                    best_accuracy[0] = accuracy
                    best_accuracy[1] = self.W
                    best_accuracy[2] = self.bias
                print("epoch", epoch, "complete with accuracy", 
                      accuracy, ".")
        self.W = best_accuracy[1]
        self.bias = best_accuracy[2]
        return (self.W, self.bias)

class AveragedPerceptron(Perceptron):
    """
    A Perceptron subclass that implements an averaged perceptron.

    This class inherits most methods from Perceptron abstract class.
    It overrides the train method.
    """
    def __init__(self, num_features, margin=.01, lr=.01,  epochs=10):
        """
        Constructor for AveragedPerceptron class.

        Accepts basic information for the simple perceptron and 
        instantiates a margin and the rest according to the 
        Perceptron abstract class. 

        Args:
            num_features: the number of data features
            margin: the perceptron margin defaults to .01
            lr: the learning rate which defaults to .01
            epochs: the number of training epochs which defaults 10
        """
        super(AveragedPerceptron, self).__init__(num_features, lr, epochs)
        
    def train(self, train_data, train_labels):
        """
        Trains an averaged perceptron on a given data and label set.

        Args:
            train_data: the train data
            train_labels: the corresponding trian label array

        Returns:
            A tuple where the first entry is the Perceptron weight and
            the second is the bias.
        """
        timestep = 1
        num_examples = train_data.shape[0]
        
        averagedW = self.W
        averagedBias = self.bias
        
        for epoch in range(self.epochs):
            misclassifications = 0
            
            for i in range(num_examples):
                example = train_data[i]
                actual_label = train_labels[i]
                predicted_label = self.W.T.dot(example) + self.bias
                
                if (actual_label * predicted_label) < 0:
                    self.W = self.W + (self.lr * actual_label \
                                       * example)
                    self.bias = self.bias + (self.lr * actual_label)
                    self.averagedW = averagedW + (self.lr \
                                                  * actual_label \
                                                  * example * timestep)
                    self.averagedBias  = averagedBias + (actual_label \
                                                         * timestep)
                    misclassifications += 1
            
            if misclassifications == 0:
                print("epoch", epoch, "completed with 100 perfect \
                      accuracy.")
                
                self.W = self.W - ((1/timestep) * averagedW)
                self.bias = self.bias - ((1/timestep) * averagedBias)
                
                return (self.W, self.bias)
            else:
                timestep += 1 
                accuracy = ((num_examples - misclassifications) \
                             / num_examples)
                self.accuracies[epoch] = accuracy
                print("epoch", epoch, "complete with accuracy", \
                      accuracy, ".")
        
        self.W = self.W - ((1/timestep) * averagedW)
        self.bias = self.bias - ((1/timestep) * averagedBias)
        
        return (self.W, self.bias)

class AggressiveMarginPerceptron(Perceptron):
    """
    A Perceptron subclass that implements an aggressive margin 
    perceptron.

    This class inherits most methods from Perceptron abstract class.
    It overrides the train method.
    """
    def __init__(self, num_features, margin=.01, lr=.01,  epochs=10):
        """
        Constructor for AggressiveMarginPerceptron class.

        Accepts basic information for the simple perceptron and 
        instantiates a margin and the rest according to the 
        Perceptron abstract class. 

        Args:
            num_features: the number of data features
            margin: the perceptron margin defaults to .01
            lr: the learning rate which defaults to .01
            epochs: the number of training epochs which defaults 10
        """
        super(AggressiveMarginPerceptron, self).__init__(num_features,
                                                         lr, epochs)
        self.margin = margin
        
    def train(self, train_data, train_labels):
        """
        Trains an aggressive margin perceptron on a given data and
        label set.

        Args:
            train_data: the train data
            train_labels: the corresponding trian label array

        Returns:
            A tuple where the first entry is the Perceptron weight and
            the second is the bias.
        """
        timestep = 1
        num_examples = train_data.shape[0]
        
        best_accuracy = [0, [], 0]
        
        for epoch in range(self.epochs):
            misclassifications = 0
            
            for i in range(num_examples):
                example = train_data[i]
                actual_label = train_labels[i]
                predicted_label = self.W.T.dot(example) + self.bias
                
                if (actual_label * predicted_label) < self.margin:
                    adjusted_lr = ((self.margin - (actual_label \
                                    * predicted_label)) \
                                    / (example.T.dot(example) + 1))
                    self.W = self.W + (adjusted_lr * actual_label \
                                       * example)
                    self.bias = self.bias + (adjusted_lr \
                                             * actual_label)
                    
                    misclassifications += 1
            
            if misclassifications == 0:
                print("epoch", epoch, "completed with 100 perfect \
                      accuracy.")
                return (self.W, self.bias)
            else:
                timestep += 1 
                accuracy = ((num_examples - misclassifications) \
                             / num_examples)
                self.accuracies[epoch] = accuracy
                if accuracy > best_accuracy[0]:
                    best_accuracy[0] = accuracy
                    best_accuracy[1] = self.W
                    best_accuracy[2] = self.bias
                print("epoch", epoch, "complete with accuracy", \
                      accuracy, ".")
        self.W = best_accuracy[1]
        self.bias = best_accuracy[2]
        return (self.W, self.bias)