###################################################
#经过对特征分析发现:性别Sex,社会阶层Pclass,
# 登船港口Embarked,年龄Age,票价Fare，
# 亲属关系（SibSp,Parch）均对存活率有影响；
###################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class LogisticRegression():

    def __init__(self,trainset,testset,alpha=0.0001,C=0,iter=500,error=0.0001):
        self.trainset = trainset
        self.testset = testset
        self.theta = np.zeros(self.trainset.shape[1]).reshape(self.trainset.shape[1],1)
        self.alpha = alpha
        self.C = C
        self.iter = iter
        self.error = error
        self.JCost = []
        self.train_X = None
        self.train_Y = None

    def DesignMatrix(self):
        '''
        构建特征矩阵
        :return: X,Y
        '''
        list = []
        [list.append(1) for item in range(len(self.trainset))]
        self.trainset.insert(0, "constant", list)
        self.train_X = self.trainset.drop(columns=['Survived'],axis=1)
        self.train_Y = self.trainset[['Survived']]
        list = []
        [list.append(1) for item in range(len(self.testset))]
        self.testset.insert(0, "constant", list)
        self.test_X = self.testset.drop(columns=['Survived'], axis=1)
        # 数据归约 : 线性模型需要用标准化的数据建模
        std = StandardScaler()
        std.fit(self.train_X)
        self.train_X = std.transform(self.train_X)
        self.train_Y = np.array(self.train_Y)
        self.test_X = std.transform(self.test_X)

    def sigmod(self,X):
        X = np.array(X,dtype=np.float32)
        h = 1 / (1 + np.exp(-X))
        return h.reshape(h.shape[0],1)

    def CostFunction(self,X,Y):
        h = self.sigmod(np.dot(X,self.theta))
        reg = self.C/len(Y)*np.sum(self.theta.T*self.theta)
        J =  -np.mean(Y * np.log(h) + (1 - Y) * np.log(1-h))+reg
        return J

    def gradientDescent(self,X,Y):
        h = self.sigmod(np.dot(X, self.theta))
        theta = self.theta + self.alpha*np.dot(X.T,(Y - h))+self.C/len(Y)*self.theta
        return theta

    def logRegression(self):
        iter = 0
        err = np.inf
        J = 0
        self.DesignMatrix()
        while(iter < self.iter and np.abs(err) >= self.error ):
            last_J = J
            J = self.CostFunction(self.train_X,self.train_Y)
            self.theta = self.gradientDescent(self.train_X,self.train_Y)
            err = J - last_J
            self.JCost.append(J)

            iter +=1
        self.plotJCost()

        h = self.predict(self.test_X)
        return h

    def plotJCost(self):
        plt.plot(self.JCost)
        # plt.savefig('logRegression.jpg')
        plt.show()

    def predict(self,textset):
        h = self.sigmod(np.dot(textset, self.theta))
        for i in range(len(h)):
            if(h[i] <= 0.5):
                h[i] = 0
            else:
                h[i] = 1
        return np.array(h,dtype=int)

    def score(self,Y_text,h):
        #真实正例
        truePositives = np.sum(Y_text)
        #真实负例
        trueNegatives = len(Y_text) - truePositives
        #预测正例
        errorPositives = 0
        errorNegatives = 0
        rightPositives = 0
        rightNegatives = 0

        for i in range(len(Y_text)):
            if(Y_text[i] == 0 and h[i] == 1):
                    errorPositives += 1
            elif(Y_text[i] == 1 and h[i] == 1):
                    rightPositives +=1
            elif(Y_text[i]== 1 and h[i] == 0):
                    errorNegatives += 1
            else:
                    rightNegatives +=1
        TP = rightPositives
        FP = errorPositives
        TN = rightNegatives
        FN = errorNegatives
        Accuracy = (TP+TN)/(TP+TN+FP+FN)
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1Score = 2*Precision*Recall/(Precision+Recall)
        print("Accuracy = ",Accuracy)
        print("Precision = ",Precision)
        print("Recall = ",Recall)
        print("F1Score = ",F1Score)