import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools

class KNN():
    def __init__(self):
        self.k = 10  # You can choose a different value
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
    
    def fit(self, x_train, y_train):
        self.knn.fit(x_train, y_train)

    def predict(self, x_test, y_test):
        y_pred = self.knn.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    
    def perform_classification(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

        accuracies= {}
        for channel in range(x.shape[1]):
            accuracies[channel] = 0

        # Generate all unique pairs
        pairs = list(itertools.combinations(range(10), 2))
        for pair in pairs:
            train_indices = (y_train == pair[0]) | (y_train == pair[1])
            test_indices = (y_test == pair[0]) | (y_test == pair[1])

            x_train_temp = x_train[train_indices]
            x_test_temp = x_test[test_indices]
            y_train_temp = y_train[train_indices]
            y_test_temp = y_test[test_indices]

            for channel in range(x.shape[1]):
                channel_train = x_train_temp[:,channel,:]
                channel_test = x_test_temp[:,channel,:]
                self.fit(channel_train, y_train_temp)
                accuracy = self.predict(channel_test, y_test_temp)
                accuracies[channel] += accuracy
                #print("pair: (%d,%d), channel %d: %s" % (pair[0], pair[1], channel, accuracy))
        
        for channel in range(x.shape[1]):
            accuracies[channel] = accuracies[channel]/len(pairs)
        
        accuracies = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)
        print(accuracies)