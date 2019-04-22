# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:09:48 2019


"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df_gender=pd.read_csv(open('gender_submission.csv'))
df_test=pd.read_csv(open('test.csv'))
df_train=pd.read_csv(open('train.csv'))
print(df_train)
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

df_temAge=df_train[['Age','Survived','Fare','Parch','SibSp','Pclass']]
df_temAgeNotNull=df_temAge.loc[(df_train['Age'].notnull())]
df_temAgeIsNull=df_temAge.loc[(df_train['Age'].isnull())]
x=df_temAgeNotNull.values[:,1:]
y=df_temAgeNotNull.values[:,0]
rfr=RandomForestRegressor(n_estimators=1000,n_jobs=-1)
rfr.fit(x,y)
predictAges=rfr.predict(df_temAgeIsNull.values[:,1:])
df_train.loc[df_train['Age'].isnull(),['Age']]=predictAges

trainSurvivedSex=df_train.loc[:,['Survived','Sex']]
listSurvivedSex=trainSurvivedSex.groupby('Sex').sum()
listTotalSex=trainSurvivedSex.groupby('Sex').count()
trainSurvivedSexFemale=trainSurvivedSex[~trainSurvivedSex['Sex'].isin(['male'])]
trainSurvivedSexMale=trainSurvivedSex[~trainSurvivedSex['Sex'].isin(['female'])]
listSurvivedSexFemale=trainSurvivedSexFemale.groupby('Survived').count()
listSurvivedSexMale=trainSurvivedSexMale.groupby('Survived').count()


trainSurvivedSex=df_train.loc[:,['Survived','Age']]#Age relation
listSurvivedSex=trainSurvivedSex.groupby('Survived').sum()



font={'family':'SimHei'}
matplotlib.rc('font',**font)
plt.title("船上总人口的男女占比")
plt.pie(listTotalSex,labels=['female','male'],autopct='%.3f%%')
plt.show()
font={'family':'SimHei'}
matplotlib.rc('font',**font)
plt.title("女性存活占比")
plt.pie(listSurvivedSexFemale,labels=['Death','Survived'],autopct='%.3f%%')
plt.show()
font={'family':'SimHei'}
matplotlib.rc('font',**font)
plt.title("男性存活占比")
plt.pie(listSurvivedSexMale,labels=['Death','Survived'],autopct='%.3f%%')
plt.show()
font={'family':'SimHei'}
matplotlib.rc('font',**font)
plt.title("生存总人口的男女占比")
plt.pie(listSurvivedSex,labels=['female','male'],autopct='%.3f%%')
plt.show()



fig=plt.figure()
from pylab import *
subplots_adjust(left=0.0,bottom=0.0,top=1,right=2,hspace=0.2,wspace=0.2)
fig.add_subplot(121)
x=df_train['Age'].hist(bins=100)
plt.xlabel("年龄")
plt.ylabel("存活人数")
plt.title("幸存者人数的年龄分布")
plt.show()



cls=[0,12,18,35,50,60,100]
df_train['Age_group']=pd.cut(df_train['Age'],cls)
trainSurvivedAge0=df_train.Age_group[df_train.Survived==0].value_counts()
trainSurvivedAge1=df_train.Age_group[df_train.Survived==1].value_counts()
data_pic=pd.DataFrame({"幸存":trainSurvivedAge1,"未幸存":trainSurvivedAge0})
data_pic.plot(kind='bar')
plt.title('各年龄段的幸存情况')
plt.show()

tem=df_train.groupby('Age_group')['Survived'].mean()
tem.plot(kind='bar',title='各年龄段的幸存概率')
plt.ylabel("")
plt.show()



df_train.groupby('SibSp')['Survived'].count().plot(kind='bar'
                ,title='弟兄姐妹及配偶个数的分布')
plt.xlabel("弟兄姐妹及配偶个数")
plt.ylabel("人数")
plt.show()
df_train.groupby('SibSp')['Survived'].mean().plot(kind='bar'
                ,title='弟兄姐妹及配偶个数与幸存的关系')
plt.xlabel("弟兄姐妹及配偶个数")
plt.ylabel("存活概率")
plt.show()




df_train.groupby('Parch')['Survived'].count().plot(kind='bar'
                ,title='父母及子女个数的分布')
plt.xlabel("父母及子女个数")
plt.ylabel("人数")
plt.show()
df_train.groupby('Parch')['Survived'].mean().plot(kind='bar'
                ,title='父母及子女个数与幸存的关系')
plt.xlabel("父母及子女个数")
plt.ylabel("存活概率")
plt.show()



df_train.groupby('Pclass')['Survived'].count().plot(kind='bar'
                ,title='舱位人数分布')
plt.xlabel("舱位等级")
plt.ylabel("人数")
plt.show()
df_train.groupby('Pclass')['Survived'].mean().plot(kind='bar'
                ,title='舱位与存活率')
plt.xlabel("舱位等级")
plt.ylabel("存活概率")
plt.show()


df_train.groupby('Embarked')['Survived'].count().plot(kind='bar'
                ,title='登船口人数分布')
plt.xlabel("登船口")
plt.ylabel("人数")
plt.show()
df_train.groupby('Embarked')['Survived'].mean().plot(kind='bar'
                ,title='登船口与幸存概率关系')
plt.xlabel("登船口")
plt.ylabel("存活概率")
plt.show()


fig=plt.figure()
from pylab import *
subplots_adjust(left=0.0,bottom=0.0,top=1,right=2,hspace=0.2,wspace=0.2)
fig.add_subplot(121)
x=df_train['Fare'].hist(bins=70)
plt.xlabel("票价")
plt.ylabel("存活人数")
plt.title("幸存者人数的票价分布")
plt.show()


cls=[0,30,50,100,200,300,600]
df_train['Fare_group']=pd.cut(df_train['Fare'],cls)
trainSurvivedFare0=df_train.Fare_group[df_train.Survived==0].value_counts()
trainSurvivedFare1=df_train.Fare_group[df_train.Survived==1].value_counts()
data_pic=pd.DataFrame({"幸存":trainSurvivedFare1,"未幸存":trainSurvivedFare0})
data_pic.plot(kind='bar')
plt.title('票价与幸存关联')
plt.show()

tem=df_train.groupby('Fare_group')['Survived'].mean()
tem.plot(kind='bar',title='各票价段的幸存概率')
plt.ylabel("")
plt.show()

