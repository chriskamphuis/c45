import numpy as np
import math


class Node:
    """
    This class represents the nodes of the tree. These nodes can either be
    leafes or have childs.
    """
    def __init__(self, X, y, mostPresentClassParent="", minsplit=4):
        if(len(X) != len(y)):
            raise("unequal length attributes + classes")
        self.classify = ""
        self.childeren = []
        self.mostPresentClass = self.mostPresent(y)
        self.toSplit = None  # Attribute childs are based on
        self.splitValue = None  # If < then this: go to left else right
        if len(X) is 0:
            self.classify = mostPresentClassParent
        elif len(X) < minsplit:
            self.classify = self.mostPresent(y)
        elif len(set(y)) is 1:
            self.mostPresentClass = self.mostPresent(y)
            self.classify = y[0]
        else:
            self.mostPresentClass = self.mostPresent(y)
            info = self.information(y)
            bestGain = 0
            bestIndex = None
            bestAttribute = None
            Xtemp = X
            ytemp = y
            i = 0
            for attribute in X.T:
                gain, index, indices = self.getGain(attribute, y, info)
                if gain > bestGain:
                    bestGain = gain
                    bestIndex = index
                    bestAttribute = i
                    if indices is not None:
                        Xtemp = X[indices]
                        ytemp = y[indices]
                i += 1
            X = Xtemp
            y = ytemp
            if bestGain >0:
                print "split on attribute {0}, at a value of {1}, with a gain of {2}".format(bestAttribute,X[bestIndex][bestAttribute],bestGain)            
                self.toSplit = bestAttribute
            elif bestGain is 0:
                print "No split had to be made"
            
            if isinstance(X[0][self.toSplit], float):
                self.splitValue = X[bestIndex][self.toSplit]
                childLeft = Node(X[bestIndex:], y[bestIndex:],
                                 self.mostPresentClass, minsplit=minsplit)
                childRight = Node(X[:bestIndex], y[:bestIndex],
                                  self.mostPresentClass, minsplit=minsplit)
                self.childeren = [childLeft, childRight]
            else:
                splits = list(set(X.T[self.toSplit]))
                for e in splits:
                    indices = [i for i, j in enumerate(X.T[self.toSplit]) if j == e]
                    self.childeren.append(Node(X[indices], y[indices]), self.mostPresentClass)
                self.classify = self.mostPresentClass

    """
    This function looks which class is most present in the trainingsdata, if
    there are less than minsplit attributes but there are more then 0
    attributes this will become the leaf label. Its value is also saved
    otherwise because it should be given to the child node if this one has no
    values.
    """
    def mostPresent(self, y):
        counter = {}
        for e in y:
            try:
                counter[str(e)] += 1
            except KeyError:
                counter[str(e)] = 1
        mostPresentClass = ""
        countMostPresentClass = 0
        for key in counter:
            if counter[key] > countMostPresentClass:
                mostPresentClass = key
                countMostPresentClass = counter[key]
        return mostPresentClass

    """
    Determine the gain when splitting on attribute X with target y. Both should
    be the same size, X should contain the sample values of that attribute
    while y should represent the class of that sample.
    """
    def getGain(self, X, y, information):
        if isinstance(X[0], str):  # Return gain value and empty sting
            coupled = np.array([list(X), list(y)]).T
            indices = np.argsort(X)
            coupled = coupled[indices]
            possibleValues = list(set(X))
            gain = information
            for i in range(len(possibleValues)):
                gain -= self.information([y for [x, y] in coupled if x is i])
            return gain, None, None
        else:  # if numeric return gain and split point (str)
            coupled = np.array([list(X), list(y)]).T
            indices = np.argsort(X)
            coupled = coupled[indices]
            targets = [b for [a, b] in coupled]
            currentBestIndex = None
            currentBestGain = 0
            for i in range(len(coupled)):
                gain = information - self.information(targets[:i])
                gain = gain - self.information(targets[i:])
                if gain > currentBestGain:
                    currentBestGain = gain
                    currentBestIndex = i
            return currentBestGain, currentBestIndex, indices

    def information(self, y):
        n = float(len(y))
        counter = {}
        for e in y:
            try:
                counter[e] += 1
            except KeyError:
                counter[e] = 1
        counts = [float(value) for value in counter.values()]
        info = 0.0
        for e in counts:
            info += float((-e/n)) * math.log(e/n, 2)
        return info

    def classifyfunction(self, x):
        if len(self.childeren) == 0:
            return self.classify
        elif isinstance(x[self.toSplit], float):
            if x[self.toSplit] > self.splitValue:
                return self.childeren[0].classifyfunction(x)
            else:
                return self.childeren[1].classifyfunction(x)
        else:
            value = x[self.toSplit]
            for child in self.childeren:
                if child.X[0][self.toSplit] == value:
                    return child.classifyfunction(x)
