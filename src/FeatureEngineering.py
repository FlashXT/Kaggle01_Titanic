###################################################
#Feature Engineering：
###################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
# pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',150)

def FeatureName(trainset):

    # 1.在数据的Name项中包含了对该乘客的称呼,将这些关键词提取出来,然后做分列处理.
    #  从名字中提取出称呼： df['Name].str.extract()是提取函数,配合正则一起使用
    Name1 = trainset['Name'].str.extract('.+,(.+)', expand=False).str.extract('^(.+?)\.', expand=False) \
            .str.strip()
    trainset.insert(0,'Name1',Name1)
    # print(trainset['Name1'])
    # 将姓名分类处理(),某些称呼可以代表职业或者身份;
    trainset['Name1'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
    trainset['Name1'].replace(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty', inplace=True)
    trainset['Name1'].replace(['Mme', 'Ms', 'Mrs'], 'Mrs', inplace=True)
    trainset['Name1'].replace(['Mlle', 'Miss'], 'Miss', inplace=True)
    trainset['Name1'].replace(['Mr'], 'Mr', inplace=True)
    trainset['Name1'].replace(['Master'], 'Master', inplace=True)

    #替换为数值型
    namelist = list(set(trainset['Name1']))
    # print(namelist)
    for name in namelist:
        trainset['Name1'].replace([name], namelist.index(name), inplace=True)


    # # 2.从姓名中提取出姓做特征
    # 从姓名中提取出姓
    Name2 = trainset['Name'].apply(lambda x: x.split('.')[1])
    trainset.insert(0, 'Name2', Name2)
    # print(trainset[['Name','Name2']])
    # 计算数量,然后合并数据集
    Name2_sum = trainset['Name2'].value_counts().reset_index()
    Name2_sum.columns = ['Name2', 'Name2_sum']

    trainset = pd.merge(trainset, Name2_sum, how='left', on='Name2')

    # 由于出现一次时该特征为无效特征,用one来代替出现一次的姓
    trainset.loc[trainset['Name2_sum'] == 1, 'Name2_new'] = 'one'
    trainset.loc[trainset['Name2_sum'] > 1, 'Name2_new'] = trainset['Name2']
    del trainset['Name2']
    # print(trainset.head(5))
    # 替换为数值型
    name2list = list(set(trainset['Name2_new']))
    for name in name2list:
        trainset['Name2_new'].replace([name], name2list.index(name), inplace=True)

    return trainset[['PassengerId','Name1','Name2_new']]

def FeatureAge(trainset):
    # 考虑年龄缺失值可能影响死亡情况,
    # print( trainset.loc[trainset["Age"].isnull()]['Survived'].mean())
    # print(trainset.loc[trainset["Age"].notnull()]['Survived'].mean())
    # 数据表明, 年龄缺失的死亡率为0.19;年龄不缺失的死亡率为0.28;所以用年龄是否缺失值来构造新特征
    trainset.loc[trainset["Age"].isnull(), "age_nan"] = 1
    trainset.loc[trainset["Age"].notnull(), "age_nan"] = 0
    trainset = pd.get_dummies(trainset, columns=['age_nan'])
    # print(trainset.head(5))
    # 创建没有['Age','Survived']的数据集
    missing_age = trainset.drop(['Survived','Cabin'], axis=1)
    # 将Age完整的项作为训练集、将Age缺失的项作为测试集。
    missing_age_train = missing_age[missing_age['Age'].notnull()]
    missing_age_test = missing_age[missing_age['Age'].isnull()]
    # 构建训练集合预测集的X和Y值
    missing_age_train_X = missing_age_train.drop(['Age'], axis=1)
    missing_age_train_Y = missing_age_train['Age']
    missing_age_test_X = missing_age_test.drop(['Age'], axis=1)
    # 先将数据标准化
    ss = StandardScaler()
    # 用测试集训练并标准化
    ss.fit(missing_age_train_X)
    missing_age_X_train = ss.transform(missing_age_train_X)
    missing_age_X_test = ss.transform(missing_age_test_X)

    # print(missing_age_X_train.shape)
    # print(missing_age_X_test.shape)
    # 使用贝叶斯预测年龄
    bayes = linear_model.BayesianRidge()
    bayes.fit(missing_age_X_train, missing_age_train_Y)
    # BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
    #               fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
    #               normalize=False, tol=0.001, verbose=False)
    # 利用loc将预测值填入数据集
    trainset.loc[(trainset['Age'].isnull()), 'Age'] = bayes.predict(missing_age_X_test)

    #分箱, 将年龄划分是个阶段10以下,10-18,18-30,30-50,50以上
    trainset['AgeDiscret'] = pd.cut(trainset['Age'], bins=[0, 10, 18, 30, 50, 100], labels=[1, 2, 3, 4, 5])

    return trainset[['PassengerId','AgeDiscret']]

def FeatureEngineer():
    trainset = pd.read_csv('.\\..\\Data\\train.csv')
    testset = pd.read_csv('.\\..\\Data\\test.csv')
    # (1)先将数据集合并,一起做特征工程(注意,标准化的时候需要分开处理)
    # 先将test补齐,然后通过pd.apped()合并
    testset['Survived'] = 0
    train_test = trainset.append(testset)

    # (2)逐个特征处理,处理复杂的特征单独编写函数

    # ① Pclass, 乘客等级, 1是最高级
    #两种方式:(比较那种方式模型效果更好, 就选那种)
    # 一是该特征不做处理, 可以直接保留.
    # 二是进行分列处理(独热编码)，可以将非数值特征转化为数值型，属于简单粗暴的方法
    # train_test = pd.get_dummies(train_test, columns=['Pclass'])

    # ② Sex, 性别 无缺失值, 直接分列(或替换为数值型)
    # train_test = pd.get_dummies(train_test, columns=["Sex"])

    train_test.loc[train_test["Sex"] == 'female','Sex'] = 1
    train_test.loc[train_test["Sex"] == 'male','Sex'] = 0

    # ③ SibSp and Parch 兄妹配偶数 / 父母子女数
    # 第一次直接保留:这两个都影响生存率,且都是数值型,先直接保存.
    train_test['SibSp_Parch'] = train_test['SibSp'] + train_test['Parch']
    del train_test['SibSp']
    del train_test['Parch']

    # ④ Embarked 数据有极少量(3个)缺失值,但是在分列的时候,缺失值的所有列可以均为0,
    # 所以可以考虑不填充.另外,也可以考虑用测试集众数来填充.先找出众数,再采用df.fillna()
    # 方法填充缺失值
    embark = train_test[['Embarked']].mode() #mode()函数计算众数
    train_test['Embarked'].fillna(embark['Embarked'][0], inplace=True)
    #替换为数值型
    train_test.loc[train_test["Embarked"] == 'C', "Embarked"] = 0
    train_test.loc[train_test["Embarked"] == 'Q', "Embarked"] = 1
    train_test.loc[train_test["Embarked"] == 'S', "Embarked"] = 2

    # train_test = pd.get_dummies(train_test, columns=["Embarked"])

    # ⑤ 处理Name特征
    NameFrame = FeatureName(train_test[['PassengerId','Name']])
    #dataframe
    train_test = pd.merge(train_test,NameFrame, how ='left', on = 'PassengerId')
    # print(train_test.head(5))
    del train_test['Name']
    # # 分列处理
    # train_test = pd.get_dummies(train_test, columns=['Name1'])
    # train_test = pd.get_dummies(train_test, columns=['Name2_new'])


    # ⑥ Fare 该特征有缺失值,先找出缺失值的那组数据,然后用平均数填充
    # 从上面的分析,发现该特征train集无miss值,test有一个缺失值,先查看
    # print(train_test.loc[train_test["Fare"].isnull()])

    # 票价与pclass和Embarked有关,所以用train分组后的平均数填充
    # print(trainset.groupby(by=["Embarked"]).Embarked.count())
    # print(trainset.groupby(by=["Pclass"]).Pclass.count())
    # print(trainset.groupby(by=["Pclass", "Embarked"]).Fare.mean()[3]['S'])
    # 用pclass=3(pclass的众数)和Embarked=S(Embarked的众数)的平均数14.644083来填充
    fare = trainset.groupby(by=["Pclass", "Embarked"]).Fare.mean()[3]['S']
    train_test["Fare"].fillna(fare, inplace=True)

    #⑦ Ticket该列和名字做类似的处理,先提取,然后分列
    # 将Ticket提取字符列
    # str.isnumeric() 如果S中只有数字字符，则返回True，否则返回False
    # train_test['Ticket_Letter'] = train_test['Ticket'].str.split().str[0]
    # train_test['Ticket_Letter'] = train_test['Ticket_Letter'].apply(lambda x: np.nan if x.isnumeric() else x)
    # train_test.drop('Ticket', inplace=True, axis=1)
    # # # # 分列,此时nan值可以不做处理
    # # train_test = pd.get_dummies(train_test, columns=['Ticket_Letter'], drop_first=True)
    del train_test['Ticket']
    # ⑧ Age
    # 1. 该列有大量缺失值,考虑用一个回归模型进行填充.
    # 2. 在模型修改的时候,考虑到年龄缺失值可能影响死亡情况,用年龄是否缺失值来构造新特征
    AgeFrame = FeatureAge(train_test)
    train_test = pd.merge(train_test, AgeFrame, how='left', on='PassengerId')
    # print(train_test.head(10))
    del train_test['Age']
    # train_test = pd.get_dummies(train_test, columns=['AgeDiscret'], drop_first=True)

    # ⑨ Cabin
    # cabin项缺失太多，直接舍去该特征
    train_test.drop(columns=['Cabin'],axis=1, inplace=True)

    #删除PassengerId 无关特征
    # del train_test['PassengerId']

    # ⑩特征处理完毕,划分数据集
    train_data = train_test[:891]
    test_data = train_test[891:]
    train_data_X = train_data.drop(['Survived'], axis=1)
    train_data_Y = train_data['Survived']
    test_data_X = test_data.drop(['Survived'], axis=1)
    train_data.to_csv('..\\Data\\DataAFE\\train_data.csv',index=False)
    test_data.to_csv('..\\Data\\DataAFE\\test_data.csv',index=False)

    # 最后还需要，数据归约
    # 1.线性模型需要用标准化的数据建模, 而树类模型不需要标准化的数据
    # 2.处理标准化的时候, 注意将测试集的数据transform到test集上


if __name__ == '__main__':

    FeatureEngineer()
