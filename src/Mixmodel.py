
import datetime

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

#模型融合 voting
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

def Voting(trainset,testset):

    lr = LogisticRegression(C=0.1,max_iter=150)
    xgb_model = xgb.XGBClassifier(n_estimators=150,max_depth=6)
    rf = RandomForestClassifier(n_estimators=200,min_samples_leaf=3,max_depth=6,oob_score=True)
    gbdt = GradientBoostingClassifier(learning_rate=0.1,min_samples_leaf=3,max_depth=6,n_estimators=200)
    vot = VotingClassifier(estimators=[('lr',lr),('rf',rf),('gbdt',gbdt),('xgb',xgb_model)],voting='hard')
    train_data_X_std, testdata_std = DataFormate(trainset, testset)
    train_data_Y = trainset['Survived']
    vot.fit(train_data_X_std,train_data_Y)
    testset['Survived'] = vot.predict(testdata_std)
    VOT = testset[['PassengerId', 'Survived']]
    VOT.to_csv('..\\Results\\VOT.csv', index=False)

#模型融合 stacking
def Stacking(trainset,testset):
    train_data_X_std, testdata_std = DataFormate(trainset, testset)
    train_data_Y = trainset['Survived']
    clfs = [LogisticRegression(C=0.1,max_iter=150),
            xgb.XGBClassifier(n_estimators=150,max_depth=6),
            RandomForestClassifier(n_estimators=200,min_samples_leaf=3,max_depth=6,oob_score=True),
            GradientBoostingClassifier(learning_rate=0.1,min_samples_leaf=3,max_depth=6,n_estimators=200)]
    #创建n_folds
    n_folds = 5

    skf = StratifiedKFold(n_splits=n_folds,shuffle=False,random_state=0)
    #第一层Stacking结果存放的位置
    train_blend = np.zeros((train_data_X_std.shape[0],len(clfs)))
    test_blend = np.zeros((testdata_std.shape[0],len(clfs)))
    #对于每一个模型，都要进行一次stacking
    for modelj,clf in enumerate(clfs):
        # modelj进行stacking
        print("Model:",modelj)
        test_blend_modelj = np.zeros((testdata_std.shape[0],n_folds))
        i = 0
        for train,test in skf.split(train_data_X_std,train_data_Y):
            print("\tFold:",i,"...")
            X_train,Y_train,X_test,Y_test = train_data_X_std[train],train_data_Y[train],train_data_X_std[test],train_data_Y[test]

            clf.fit(X_train,Y_train)
            # 每一折都要对整个训练集中test那份进行预测
            #predict_proba返回（n，j）的数据，第j列表示预测标签为j的概率
            y = clf.predict_proba(X_test)
            train_blend[test,modelj] = y[:,1]
            #同时对测试集进行预测
            test_blend_modelj[:,i] = clf.predict_proba(testdata_std)[:,1]
            i+=1
        #对每一折对测试集的的预测结果求均值
        test_blend[:,modelj] = test_blend_modelj.mean(1)
    #建立第二层模型
    #使用GridSearchCV调参
    # param = {'C':[0.001,0.01,0.1,1,10],"max_iter":[100,200,300,500]}
    # lr = LogisticRegression()
    # clf = GridSearchCV(lr,param,cv=5,n_jobs=-1,verbose=1,scoring='roc_auc')
    # clf.fit(train_blend,train_data_Y)
    #
    # print(clf.best_params_)
    clf = LogisticRegression(C=0.1,max_iter=100)
    clf.fit(train_blend, train_data_Y)
    testset['Survived'] = clf.predict(test_blend)
    stack = testset[['PassengerId', 'Survived']]
    stack.to_csv('..\\Results\\mystack.csv', index=False)


def main():
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%S:%M"))
    trainset = pd.read_csv("..\\Data\\DataAFE\\train_data.csv")
    testset = pd.read_csv("..\\Data\\DataAFE\\test_data.csv")
    # Voting(trainset,testset)
    Stacking(trainset,testset)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%S:%M"))
if __name__ == "__main__":
    main()