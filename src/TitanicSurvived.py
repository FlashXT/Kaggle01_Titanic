import pandas as pd
import datetime
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src import LogisticRegression as LR

def RandomForest(trainset,testset):
    #class sklearn.ensemble.RandomForestClassifier(
    #       n_estimators=’warn’,    随机森林中数的数目,默认为10；
    #       criterion=’gini’,       特征选择标准可以使用"gini"或者"entropy"，默认的基尼系数"gini"；
    #       max_depth=None,
    #       min_samples_split=2,
    #       min_samples_leaf=1,
    #       min_weight_fraction_leaf=0.0, 叶子结点样本权重和的最小值，如果小于这个值，则会和兄弟节点
    #                                     一起被剪枝。 默认是0，就是不考虑权重问题。
    #       max_features=’auto’,    划分的最大特征数，默认auto,为输入特征数的平方根sqrt(N)，
    #                               该参数可取值 “auto”,max_features=sqrt(n_features);
    #                                          “sqrt”,max_features=sqrt(n_features)
    #                                          “log2”,max_features=log2(n_features)
    #                                           None, max_features=n_features.
    #       max_leaf_nodes=None,
    #       min_impurity_decrease=0.0,
    #       min_impurity_split=None,
    #       bootstrap=True,
    #       oob_score=False,
    #       n_jobs=None,
    #       random_state=None,
    #       verbose=0,
    #       warm_start=False,
    #       class_weight=None)

    rf = RandomForestClassifier(n_estimators= 150,min_samples_leaf=3,max_depth=15,max_features=None,oob_score=True)
    train_data_X = trainset.drop(['PassengerId','Survived'], axis=1)
    train_data_Y = trainset['Survived']
    rf.fit(train_data_X,train_data_Y)
    print(rf.oob_score_)
    # print(rf.decision_path(train_data_X.iloc[0]))
    testset["Survived"] = rf.predict(testset.drop(['PassengerId','Survived'], axis=1))

    RF = testset[['PassengerId','Survived']]
    RF.to_csv('..\\Results\\RF.csv',index=False)

def LogisticReression(trainset,testset):
    '''
    class sklearn.linear_model.LogisticRegression(
                 penalty=’l2’,
                 dual=False,
                 tol=0.0001,
                 C=1.0,
                 fit_intercept=True,
                 intercept_scaling=1,
                 class_weight=None,
                 random_state=None,
                 solver=’warn’,
                 max_iter=100,
                 multi_class=’warn’,
                 verbose=0,
                 warm_start=False,
                 n_jobs=None
    )
    :param trainset:
    :param testset:
    :return:
    '''
    lr = LogisticRegression(penalty='l2',solver='liblinear',max_iter=1500)
    train_data_X = trainset.drop(['PassengerId', 'Survived'], axis=1)
    train_data_Y = trainset['Survived']
    lr.fit(train_data_X, train_data_Y)

    testset["Survived"] = lr.predict(testset.drop(['PassengerId', 'Survived'], axis=1))

    LR = testset[['PassengerId', 'Survived']]
    LR.to_csv('..\\Results\\LR.csv', index=False)

def MyLogisticRegression(trainset,testset):
    lr = LR.LogisticRegression(trainset,testset)
    test_Y = lr.logRegression()
    testset["Survived"] = test_Y
    testset.to_csv("..\\Results\\myLR.csv",columns = ['PassengerId','Survived'],index=False)

def DecisionTree(trainset,testset):

    '''
    class sklearn.tree.DecisionTreeClassifier(
                        criterion=’gini’, 特征选择标准
                        splitter=’best’,  特征划分点选择标准(best,random)
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1, 叶子节点最少样本数
                        min_weight_fraction_leaf=0.0,
                        max_features=None,  划分的最大特征数，默认auto,为输入特征数的平方根sqrt(N)，
                                             该参数可取值 “auto”,max_features=sqrt(n_features);
                                                        “sqrt”,max_features=sqrt(n_features)
                                                        “log2”,max_features=log2(n_features)
                                                         None, max_features=n_features.
                        random_state=None,
                        max_leaf_nodes=None,
                        min_impurity_decrease=0.0,
                        min_impurity_split=None, c
                        lass_weight=None,
                        presort=False
    )
    :param trainset:
    :param testset:
    :return:
    '''
    train_data_X = trainset.drop(['PassengerId', 'Survived'], axis=1)
    train_data_Y = trainset['Survived']
    clf = tree.DecisionTreeClassifier(max_features=None,max_depth=15,min_samples_leaf=3)
    clf.fit(train_data_X,train_data_Y)
    testset["Survived"] = clf.predict(testset.drop(['PassengerId', 'Survived'], axis=1))

    CLF = testset[['PassengerId', 'Survived']]
    CLF.to_csv('..\\Results\\CLF.csv', index=False)




if __name__ == '__main__':
    trainset = pd.read_csv("..\\Data\\DataAFE\\train_data.csv")
    testset = pd.read_csv("..\\Data\\DataAFE\\test_data.csv")

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%S:%M"))
    #随机森林
    # RandomForest(trainset,testset)
    #Sklearn逻辑回归
    # LogisticReression(trainset, testset)
    #自己实现的逻辑回归
    # MyLogisticRegression(trainset, testset)
    #决策树
    DecisionTree(trainset, testset)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%S:%M"))