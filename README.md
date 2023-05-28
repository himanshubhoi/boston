# boston



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


from sklearn.datasets import load_boston
boston = load_boston()


df = pd.DataFrame(boston.data)

df.head()

df.tail()

df.columns = boston.feature_names
df

df['Price'] = boston.target

df

df.shape

df.isnull().sum()

X = df.drop(['Price'],axis=1)
y = df['Price']


y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)


from sklearn.linear_model import LinearRegression
Lm = LinearRegression()


Lm.fit(X_train,y_train)


Lm.intercept_


y_pred = Lm.predict(X_train)

y_pred

print('R^2:',metrics.r2_score(y_train, y_pred))


print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))


plt.scatter(y_train, y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()
