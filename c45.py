from sklearn import datasets
from node import Node
import numpy as np
from sklearn import cross_validation as skcv
import matplotlib.pyplot as plt


class C45:

    def __init__(self, useCrossVal, minsplit, X=None, y=None):
        self.tree = None
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        if useCrossVal:
            if X is None or y is None:
                self.X, self.X_test, self.y, self.y_test = self.getkFoldIrisData()
            else:
                self.X = X
                self.y = y
        else: 
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
        
    def getkFoldIrisData(self):
        X,y = self.getIrisData()
        X_train, X_test, y_train, y_test = skcv.train_test_split(X,y)
        return X_train, X_test, y_train, y_test


    def fit(self):
        self.tree = Node(self.X, self.y, "a", minsplit=minsplit)

    def classify(self, x):
        if len(x) != len(self.X[0]):
            raise("wrong amount attributes given")
        else:
            return self.tree.classifyfunction(x)

if __name__ == "__main__":
    useCrossVal = True
    minsplit = 20
    
    
   # print c45.classify([5.1, 3.5, 1.4, 0.2]) # Should print 0
   # print c45.classify([7.0, 3.2, 4.7, 1.4]) # Should print 1
   # print c45.classify([6.4, 2.8, 5.6, 2.2]) # Should print 2
    n=20
    a=np.empty(n)
    avg=np.empty(minsplit-1) 
    ind=0   
    for m in range(1,minsplit,1):
        a=0*a
        for j in range(0,n):
            c45 = C45(useCrossVal, m)
            y = c45.y_test
            X = c45.X_test
            for i in range(len(y)):
                if y[i] != int(c45.classify(X[i])):
                    a[j]+=1  # Should print 1
        avg[ind]=np.average(a)
        print "On average, {0} out of {1} data points were misclassified.".format(avg[ind], len(y))
        ind+=1
    plt.plot( range(1,minsplit), avg)
    plt.xlabel("minsplit")
    plt.ylabel("Average misclassifications")
    plt.legend()
    plt.show()
