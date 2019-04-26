###################################################
#经过对特征分析发现:性别Sex,社会阶层Pclass,
# 登船港口Embarked,年龄Age,票价Fare，
# 亲属关系（SibSp,Parch）均对存活率有影响；
###################################################

import numpy as np
import pandas as pd
data = pd.read_csv(".\\..\\Data\\train.csv")
SurviveAll  = len(data[data['Survived']==1])
def Sex():
    # 性别比例
    # 将sex替换为数字
    data.loc[data['Sex'] == 'male', 'Sex'] = 1
    data.loc[data['Sex'] == 'female', 'Sex'] = 0
    # 男性
    male = data[data["Sex"] == 1][['Sex', 'Survived']]
    maleSurvived = len(male[male['Survived'] == 1])
    maleDead = len(male[male['Survived'] == 0])
    maleSurviveRate = round(maleSurvived / len(male), 2)
    maleSurviveRateAll = round(maleSurvived / SurviveAll, 2)
    # 女性
    female = data[data["Sex"] == 0][['Sex', 'Survived']]
    femaleSurvived = len(female[female['Survived'] == 1])
    femaleDead = len(female[female['Survived'] == 0])
    femaleSurviveRate = round(femaleSurvived / len(female), 2)
    femaleSurviveRateAll = round(femaleSurvived /SurviveAll, 2)
    print("Sex:")
    print('Male:\t', len(male), '\tSurvived:', maleSurvived, '\tDead:',maleDead, '\tNumRate:',
          round(len(male)/len(data),2),'\tSurvivedRate:', maleSurviveRate,'SurvivedRateAll:\t', maleSurviveRateAll)
    print('Female:\t', len(female), '\tSurvived:', femaleSurvived,'\tDead:', femaleDead, '\tNumRate:',
          round(len(female)/len(data),2),'\tSurvivedRate:',femaleSurviveRate,'SurvivedRateAll:\t', femaleSurviveRateAll)
    print("===========================================================================================================")

def Pclass():

    # 社会阶级
    pclass = 1
    print("PClass:")
    while(pclass <4):

        pClass = data[data["Pclass"] == pclass][['Pclass','Survived']]
        rate = round(len(pClass) / len(data), 2)
        pClassSurvived = len(pClass[pClass['Survived'] == 1])
        pClassDead = len(pClass[pClass['Survived'] == 0])
        pClassSurviveRate = round(pClassSurvived / len(pClass), 2)
        pClassSurviveRateAll = round(pClassSurvived / SurviveAll, 2)
        print('pClass',pclass,':', len(pClass),'\tSurvived:', pClassSurvived, '\tDead:', pClassDead,  '\tNumRate:',rate,
              '\tSurvivedRate:',pClassSurviveRate,'\tSurviveRateAll:',pClassSurviveRateAll)

        pclass +=1
    print("===========================================================================================================")

def Age():
    #有缺失值,处理缺失值:填充出现概率最大的值
    age0 = data[data['Age'] <= 12][['Age', 'Survived']]
    age1 = data.loc[(data['Age'] > 12) & (data['Age'] <= 18), ('Age', 'Survived')]
    age2 = data.loc[(data['Age'] > 18) & (data['Age'] <= 60), ('Age', 'Survived')]
    age3 = data.loc[(data['Age'] > 60), ('Age', 'Survived')]
    agelist = [len(age0), len(age1), len(age2), len(age3)]
    loc = -1
    val = 0
    for key,value in enumerate(agelist):
        if value > val:
            val = value
            loc = key
    if loc == 0:
        age = 6
    elif loc == 1:
        age = 15
    elif loc == 2:
        age = 39
    else:
        age = 80
    data['Age'].fillna(age, inplace=True)

    # 离散化(分箱)，age分段为(0,12],(12,18],(18,60],(60,100]
    data.loc[data['Age'] <= 12, 'Age'] = 0
    data.loc[(data['Age'] > 12) & (data['Age'] <= 18), 'Age'] = 1
    data.loc[(data['Age'] > 18) & (data['Age'] <= 60), 'Age'] = 2
    data.loc[data['Age'] > 60, 'Age'] = 3
    age0 = data[data['Age'] == 0][['Age', 'Survived']]
    age1 = data[data['Age'] == 1][['Age', 'Survived']]
    age2 = data[data['Age'] == 2][['Age', 'Survived']]
    age3 = data[data['Age'] == 3][['Age', 'Survived']]
    agelist = [age0,age1,age2,age3]
    print('Age:')
    for age in agelist:
        ageSurvived = len(age[age['Survived'] == 1])
        ageDead = len(age[age['Survived'] == 0])
        ageSurviveRate = round(ageSurvived / len(age), 2)
        ageSurviveRateAll = round(ageSurvived / SurviveAll, 2)
        print('Age:', len(age), '\tSurvived:', ageSurvived, '\tDead:',ageDead, '\tNumRate:',
              round(len(age)/len(data),2),'\tSurvivedRate:',ageSurviveRate,'\tSurviveRateAll:',ageSurviveRateAll)
    print("===========================================================================================================")

def SibSp():
    # 是否有亲属
    sibsp0 = data[data["SibSp"] == 0][['Sex','SibSp', 'Survived']]
    sibsp1 = data[data["SibSp"] > 0][['Sex','SibSp', 'Survived']]

    sibsp = [sibsp0,sibsp1]

    print("SibSp:")
    for item in sibsp:

        malesibSp = item[item['Sex'] == 1][['Survived']]
        malesibspSurvived = len(malesibSp[malesibSp['Survived'] == 1])
        malesibspDead = len(malesibSp[malesibSp['Survived'] == 0])
        malesibspSurviveRate = round(malesibspSurvived / len(malesibSp), 2)
        malesibspSurviveRateAll = round(malesibspSurvived / SurviveAll, 2)
    #
        femalesibSp = item[item['Sex'] == 0][['Survived']]
        femalesibspSurvived = len(femalesibSp[femalesibSp['Survived'] == 1])
        femalesibspDead = len(femalesibSp[femalesibSp['Survived'] == 0])
        femalesibspSurviveRate = round(femalesibspSurvived / len(femalesibSp), 2)
        femalesibspSurviveRateAll = round(femalesibspSurvived / SurviveAll, 2)
        print("SibSp = ",item.reset_index()['SibSp'][0],',NUM:',len(item))

        print('Male:\t', len(malesibSp), '\tSurvived:', malesibspSurvived,'\tDead:',malesibspDead,
              'Rate:',round(len(malesibSp)/len(item),2),'\tSurvivedRate:', malesibspSurviveRate,
              '\tSurviveRateAll',malesibspSurviveRateAll)
        print('Female:\t', len(femalesibSp), '\tSurvived:',femalesibspSurvived,'\tDead:', femalesibspDead,
              'Rate:',round(len(femalesibSp)/len(item),2),'\tSurvivedRate:', femalesibspSurviveRate,
              '\tSurviveRateAll', femalesibspSurviveRateAll)

    print("===========================================================================================================")

def Parch():
    # 没有孩子
    parch0 = data[data["Parch"] == 0][['Sex','Parch', 'Survived']]
    #有孩子
    parch1 = data[data["Parch"] > 0][['Sex','Parch', 'Survived']]
    parchsur = data[data['Survived'] == 1]
    Parch = [parch0,parch1]

    print("Parch:")
    for item in Parch:
        parch = len(item)
        parchSurvived = len(item[item['Survived'] == 1])
        parchDead = len(item[item['Survived'] == 0])
        parchSurviveRate = round(parchSurvived / len(item), 2)
        print("Parch = ", item.reset_index()['Parch'][0])
        print('parch:\t',parch,'\tSurvived:', parchSurvived,'\tDead:',parchDead, 'Rate:',
              round(parchSurvived/len(parchsur),2), '\tSurvivedRate:', parchSurviveRate)

    print("===========================================================================================================")

def Fare():
    fare = data[['Fare', 'Survived']]
    # 票价归一化处理
    # faremax = fare.loc[:,'Fare'].max()
    # faremin = fare.loc[:, 'Fare'].min()
    # gap =  faremax - faremin
    # fare.loc[fare['Fare'] != -1,'Fare'] = fare['Fare']/gap

    fareSurvived = fare[fare['Survived'] == 1]
    fareDead = fare[fare['Survived'] == 0]
    fareSurvivedMean = round(fareSurvived[['Fare']].mean(0)[0],2)
    fareSurvivedStd = round(np.std(fareSurvived[['Fare']])[0],2)
    fareDeadMean = round(fareDead[['Fare']].mean(0)[0], 2)
    fareDeadStd = round(np.std(fareDead[['Fare']])[0], 2)

    print('Fare:')
    print("FareSurvivedMean:", fareSurvivedMean,"\tFareSurvivedStd",fareSurvivedStd,"\tFareDeadMean:",fareDeadMean,
          "\tFareDeadStd",fareDeadStd )
    print("===========================================================================================================")

def Embarked():
    #将港口替换为数字
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'S', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2
    embarked = data[['Embarked','Survived']]
    emSurvived = embarked[embarked['Survived']==1]
    emC = embarked.loc[embarked['Embarked'] == 0]
    emS = embarked.loc[embarked['Embarked'] == 1]
    emQ = embarked.loc[embarked['Embarked'] == 2]
    rateC = round(len(emC) / len(data),2)
    rateS = round(len(emS) / len(data),2)
    rateQ = round(len(emQ) / len(data),2)

    emSurvivedC = emSurvived.loc[emSurvived['Embarked'] == 0]
    emSurvivedS = emSurvived.loc[emSurvived['Embarked'] == 1]
    emSurvivedQ = emSurvived.loc[emSurvived['Embarked'] == 2]

    emDead = embarked[embarked['Survived'] == 0]
    emDeadC = emDead.loc[emDead['Embarked'] == 0]
    emDeadS = emDead.loc[emDead['Embarked'] == 1]
    emDeadQ = emDead.loc[emDead['Embarked'] == 2]
    emSurvivedCRate = round(len(emSurvivedC)/len(emC),2)
    emSurvivedSRate = round(len(emSurvivedS) / len(emS), 2)
    emSurvivedQRate = round(len(emSurvivedQ) / len(emQ), 2)
    print("Embarked:")
    print("Embarked_C:","\tnum:",len(emC),'\tSurvived:',len(emSurvivedC),'\tDead:',len(emDeadC),'\tRate:',rateC,
          '\tSurvivedRate:',emSurvivedCRate)
    print("Embarked_S:", "\tnum:", len(emS), '\tSurvived:', len(emSurvivedS), '\tDead:', len(emDeadS),'\tRate:',rateS,
          '\tSurvivedRate:',emSurvivedSRate)
    print("Embarked_Q:", "\tnum:", len(emQ), '\tSurvived:', len(emSurvivedQ), '\tDead:', len(emDeadQ),'\tRate:',rateQ,
          '\tSurvivedRate:',emSurvivedQRate)
    print("===========================================================================================================")

if __name__ == "__main__":
    Sex()
    Pclass()
    Age()
    SibSp()
    Parch()
    Fare()
    Embarked()
    print(data.columns.values)
