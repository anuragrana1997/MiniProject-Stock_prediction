import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Tesla.csv')
print(df.head())

plt.figure(figsize=(10,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

# print(df[df['Close'] == df['Adj Close']].shape)
df = df.drop(['Adj Close'], axis=1)
# print(df.head())

# print(df.isnull().sum())

features = ["Open", "High", "Low", "Close", "Volume"]

plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sb.distplot(df[col])
plt.show()

plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sb.boxplot(x = df[col])
plt.show()

splitDate = df["Date"].str.split("/", expand = True)
df["Month"] = splitDate[0].astype("int")
df["Day"] = splitDate[1].astype("int")
df["Year"] = splitDate[2].astype("int")
df["isQuaterMonth"] = np.where(df["Month"] % 3 == 0 , 1, 0)

dataGroupedByYear = df.drop("Date", axis=1).groupby("Year").mean()
plt.subplots(figsize=(20,10))
for i, col in enumerate(["Open", "High", "Low", "Close"]):
    plt.subplot(2, 2, i+1)
    dataGroupedByYear[col].plot.bar()
plt.show()

df.drop('Date', axis=1).groupby('isQuaterMonth').mean()

df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

plt.pie(df['target'].value_counts().values, 
        labels=[0, 1], autopct='%1.1f%%')
plt.show()

plt.figure(figsize=(10, 10)) 

sb.heatmap(df.drop('Date', axis=1).corr() > 0.9, annot=True, cbar=False)
plt.show()

features = df[['open-close', 'low-high', 'isQuaterMonth']]
target = df['target']

print(features)

scaler = StandardScaler()
features = scaler.fit_transform(features)

print(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

models = [LogisticRegression(), SVC(
    kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
    models[i].fit(X_train, Y_train)

    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(
    Y_train, models[i].predict_proba(X_train)[:,1]))
    print('Validation Accuracy : ', metrics.roc_auc_score(
    Y_valid, models[i].predict_proba(X_valid)[:,1]))
    print()


