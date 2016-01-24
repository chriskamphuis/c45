from sklearn import datasets
from node import Node


class C45:

    def __init__(self, X=None, y=None):
        self.tree = None
        self.X = None
        self.y = None
        if X is None or y is None:
            self.X, self.y = self.getIrisData()
        else:
            self.X = X
            self.y = y
        self.fit()

    def getIrisData(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        return X, y

    def fit(self):
        self.tree = Node(self.X, self.y)

    def classify(self, x):
        if len(x) != len(self.X[0]):
            raise("wrong amount attributes given")
        else:
            return self.tree.classifyfunction(x)

if __name__ == "__main__":
    c45 = C45()
    print c45.classify([5.1, 3.5, 1.4, 0.2]) # Should print 0
    print c45.classify([7.0, 3.2, 4.7, 1.4]) # Should print 1
    print c45.classify([6.4, 2.8, 5.6, 2.2]) # Should print 2
