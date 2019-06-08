# @Author:FlashXT;
# @Date:2019/5/11 21:32;
# @Version 1.0
# CopyRight © 2018-2020,FlashXT & turboMan . All Right Reserved.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.width', 300)
pd.set_option('display.max_columns',20)
def DataAyalysis():
    trainset = pd.read_csv('.\\..\\Data\\train.csv')

    #(1)查看数据基本特征

    #返回列名
    print(trainset.columns.values)
    #返回 每列列名，该列非nan值个数，以及该列类型
    print(trainset.info())
    # 返回数值型变量的统计量
    trainset.describe(percentiles=[0.00,0.25,0.5,0.75,1.00])
    print(trainset.describe())

    #(2)特征分析：统计学与绘图
    #目的：初步了解数据之间的相关性，为构造特征工程以及模型建立做准备
    #① 存活人数统计
    print("SurvivedCount:")
    survived_count = trainset['Survived'].value_counts()
    print(survived_count)
    #②数值型数据的协方差，corr()函数
    #相关性协方差表，corr()函数，返回结果为相关性；实际计算的是皮尔逊相关系数
    trainset_corr = trainset.drop('PassengerId',axis=1).corr()
    print("特征相关性分析：")
    print(trainset_corr)
    print("====================================================================")

    #(3)画出相关性热力图
    plt.subplots(figsize=(15,9))#调整画布大小
    sns.heatmap(trainset_corr,vmin=-1,vmax=1,annot=True,square=True)
    plt.savefig("..\\Data\\Figure\\feature_corr.jpg")
    plt.show()

    # (4)各个特征与结果的相关性分析
    print("各个特征与结果的相关性分析:")
    print("=================================")

    #① Pclass 与 Survived的相关性
    print("Pclass 的不同取值对 Survived均值 的影响:")
    print(trainset.groupby(['Pclass'])['Pclass', 'Survived'].mean())
    trainset[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()
    # plt.savefig("..\\Data\\Figure\\Pcalss_Survived.jpg")
    plt.show()
    #结果 ： Pclass 与Survived 的相关性高，保留该特征

    #② Sex 与 Survived的相关性
    print("Sex 的不同取值对 Survived均值 的影响:")
    print(trainset.groupby(['Sex'])['Sex','Survived'].mean())
    trainset[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar()
    # plt.savefig("..\\Data\\Figure\\Sex_Survived.jpg")
    plt.show()
    # 结果 ：Sex 与Survived 的相关性高，保留该特征

    # ③SibSp 与 parch 兄妹配偶数/父母子女数
    print("SibSp 的不同取值对 Survived均值 的影响:")
    print(trainset.groupby(['SibSp'])['SibSp','Survived'].mean())
    trainset[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar()
    # plt.savefig("..\\Data\\Figure\\SibSp_Survived.jpg")
    plt.show()
    print("Parch 的不同取值对 Survived均值 的影响:")
    print(trainset.groupby(['Parch'])['Parch', 'Survived'].mean())
    trainset[['Parch', 'Survived']].groupby(['Parch']).mean().plot.bar()
    # plt.savefig("..\\Data\\Figure\\Parch_Survived.jpg")
    plt.show()
    # #结果分析:这些特征与特定的值没有相关性，可以由这些独立的特征派生出一个新特征或者一组新特征

    #④Age的不同取值对Survived的影响
    # print("Age的不同取值对 Survived均值 的影响:")
    # g = sns.FacetGrid(trainset, col='Survived', size=5)
    # g.map(plt.hist, 'Age', bins=40)
    # plt.savefig("..\\Data\\Figure\\Age_Survived.jpg")
    #
    # trainset[['Age', 'Survived']].groupby(['Age']).mean().plot()
    # plt.savefig("..\\Data\\Figure\\Age_Survived2.jpg")
    # plt.show()
    #结果: Age 对Survived 有影响,老人和小孩的存活率高,中青年存活率较低;

    #⑤Embarked登船港口与Survived相关性分析
    sns.countplot('Embarked', hue='Survived', data=trainset)
    # plt.savefig("..\\Data\\Figure\\Embarked_Survived.jpg")
    plt.show()
    #结果:不同港口登船的死亡人数与存活人数比例不一样,Embarked与 Survived相关;

    #⑥ 其他因素
    # 在数据的Name项中包含了对该乘客的称呼，如Mr、Miss等，这些信息包含了乘客的年龄、
    # 性别、也有可能包含社会地位，如Dr、Lady、Major、Master等称呼。这一项不方便用
    # 图表展示，但是在特征工程中，需要将其提取出来,然后放到模型中。
    # 剩余因素还有船票价格、船舱号和船票号，这三个因素都可能会影响乘客在船中的位置从
    # 而影响逃生顺序，但是因为这三个因素与生存之间看不出明显规律，所以在后期模型融合
    # 时，将这些因素交给模型来决定其重要性。

    #分析结果:Pclass,Sex,Embarked,Fare为首要考虑因素;Name,票价，船舱，船票号为次要因素;

if __name__ == "__main__":
    DataAyalysis()