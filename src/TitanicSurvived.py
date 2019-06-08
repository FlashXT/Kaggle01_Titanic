import pandas as pd
import datetime
from sklearn import svm
from sklearn import tree
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.Test import LogisticRegression as LR


def DataFormate(traindata,testdata):
    # 数据归约
    # 1.线性模型需要用标准化的数据建模, 而树类模型不需要标准化的数据
    # 2.处理标准化的时候, 注意将测试集的数据transform到test集上
    train_data_X = traindata.drop(['PassengerId','Survived'],axis=1)
    test_data = testdata.drop(['PassengerId','Survived'], axis=1)
    ss2 = StandardScaler()
    ss2.fit(train_data_X)
    train_data_X_std = ss2.transform(train_data_X)
    test_data_X_std = ss2.transform(test_data)
    return train_data_X_std,test_data_X_std

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
    train_data_X_std,testdata_std = DataFormate(trainset,testset)
    train_data_Y = trainset['Survived']
    lr = LogisticRegression(penalty='l2',solver='liblinear',max_iter=1500)

    lr.fit(train_data_X_std, train_data_Y)

    testset["Survived"] = lr.predict(testdata_std)

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

def SVM(trainset,testset):
    param = {'C':[0.001,0.01,0.1,1,10],"max_iter":[100,200,300]}
    train_data_X_std, testdata_std = DataFormate(trainset, testset)
    train_data_Y = trainset['Survived']
    svc = svm.SVC()
    clf = GridSearchCV(svc,param,cv=5,n_jobs=-1,verbose=1,scoring='roc_auc')

    clf.fit(train_data_X_std,train_data_Y)
    print(clf.best_params_)
    svc = svm.SVC(C=1,max_iter=300)
    #训练模型，并预测结果
    svc.fit(train_data_X_std,train_data_Y)
    testset['Survived']=svc.predict(testdata_std)
    SVM = testset[['PassengerId', 'Survived']]
    SVM.to_csv('..\\Results\\SVM.csv', index=False)

def GBDT(trainset,testset):
    gbdt = GradientBoostingClassifier(learning_rate=0.7,max_depth=6,n_estimators=100,min_samples_leaf=3)
    train_data_X = trainset.drop(['PassengerId', 'Survived'], axis=1)
    train_data_Y = trainset['Survived']
    test_data_X = testset.drop(['PassengerId', 'Survived'], axis=1)
    gbdt.fit(train_data_X,train_data_Y)
    testset["Survived"] = gbdt.predict(test_data_X)

    GBDT = testset[['PassengerId', 'Survived']]
    GBDT.to_csv('..\\Results\\GBDT.csv', index=False)

def Xgboost(trainset,testset):
    xgb_model = xgb.XGBClassifier(n_estimators=150,max_depth=6)
    train_data_X = trainset.drop(['PassengerId', 'Survived'], axis=1)
    train_data_Y = trainset['Survived']
    test_data_X = testset.drop(['PassengerId', 'Survived'], axis=1)
    xgb_model.fit(train_data_X,train_data_Y)
    testset["Survived"] = xgb_model.predict(test_data_X)

    XGB = testset[['PassengerId', 'Survived']]
    XGB.to_csv('..\\Results\\XGB.csv', index=False)


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
    # DecisionTree(trainset, testset)
    #SVM
    # SVM(trainset,testset)
    #GBDT
    # GBDT(trainset, testset)
    #XGBoost
    Xgboost(trainset, testset)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%S:%M"))