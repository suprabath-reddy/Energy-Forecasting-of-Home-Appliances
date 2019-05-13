"""S Suprabath Reddy
EE15BTECH11026
"""

"""Python 3.5"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


train = pd.read_csv("training_data.csv")
test = pd.read_csv("testing_data.csv")

f = open('result.csv', 'w')
f.write("Machine Learning Technique,Mean Absolute Error\n")

print("Number of instances in dataset = {}".format(train.shape[0]))
print("Total number of columns = {}".format(train.columns.shape[0]))
print("Column wise count of null values:-")
print(train.isnull().sum())

temp_cols = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"] #Columns for temperature sensors
rho_cols = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9"] #Columns for humidity sensors
weather_cols = ["T_out", "Tdewpoint", "RH_out", "Press_mm_hg", "Windspeed", "Visibility"] #Columns for weather data
target = ["Appliances"]  #Target variable column

train['date'] = pd.to_datetime(train['date'])
train['just_date'] = train['date'].dt.date

plt.plot(train['just_date'],train['Appliances'],color = 'red', linewidth=2, linestyle="-" )
plt.show()
plt.plot(train['just_date'],train['lights'],color = 'blue', linewidth=2, linestyle="-" )
plt.show()

train = train.drop(["just_date","date","lights"], axis=1)

train[temp_cols].describe()
train[rho_cols].describe()
train[weather_cols].describe()
train[target].describe()

#Temperature Sensors
temp_scatter = pd.plotting.scatter_matrix(train[temp_cols], diagonal="kde", figsize=(16, 16))

#Humidity Sensors
rho_scatter = pd.plotting.scatter_matrix(train[rho_cols], diagonal="kde", figsize=(16, 16))

#Weather Data
weather_scatter = pd.plotting.scatter_matrix(train[weather_cols], diagonal="kde", figsize=(16, 16))

#Histogram of each column
histograms = train.hist(figsize=(16, 16), bins=20)
plt.show()

plt.xlabel("Windspeed", fontsize='x-large')
plt.ylabel("Appliances", fontsize='x-large')
plt.xlim(-5, train.Windspeed.max() + 5)
plt.ylim(-250, 1200)
plt.scatter(train["Windspeed"], train["Appliances"])
plt.show()

plt.xlabel("Appliance Energy Consumption in Wh", fontsize="x-large")
plt.ylabel("No. of instances", fontsize="x-large")
train["Appliances"].hist(figsize=(16, 8), bins=100)


print("Percentage of dataset in range of 0-200 Wh")
print("{:.3f}%".format((train[train.Appliances <= 200]["Appliances"].count()*100.0) /train.shape[0]))

from scipy.stats import pearsonr

#Calculate the coefficient and p-value between T7 and T9
corr_coef, p_val = pearsonr(train["T7"], train["T9"])
print("Correlation coefficient : {}".format(corr_coef))
print("p-value : {}".format(p_val))

from itertools import combinations

for pair in combinations(train.columns, 2):
    col_1, col_2 = pair
    corr_coef, p_val = pearsonr(train[col_1], train[col_2])
    # Check for high correlation
    if corr_coef > 0.9 or corr_coef < -0.9:
        print("Column pair : {}, {}".format(*pair))
        print("Correlation coefficient : {}".format(corr_coef))
        print("p-value : {}".format(p_val))

plt.xlabel("T6", fontsize='x-large')
plt.ylabel("T_out", fontsize='x-large')

plt.xlim(int(train.T6.min()) - 2, int(train.T6.max()) + 2)
plt.ylim(int(train.T_out.min()) - 2, int(train.T_out.max()) + 2)

plt.scatter(train["T6"], train["T_out"])
plt.show()

#Starting Linear Regression

#Removing correlated features T6 and RH4

X_train = train.drop(["T6", "RH_4"], axis=1)
X_test = test.drop(["T6", "RH_4"], axis=1)

X_train.drop(["Appliances"], axis=1, inplace=True)
y_train = train["Appliances"]

X_test.drop(["Appliances", "date","lights"], axis=1, inplace=True)
y_test = test["Appliances"]

from sklearn.linear_model import LinearRegression
from sklearn import metrics

model = LinearRegression()
reg = model.fit(X_train, y_train)

y_predict = reg.predict(X_test)

error_lin = np.sqrt(metrics.mean_absolute_error(y_test,y_predict))
print (error_lin)
f.write("Linear Regression" + ',' + str(error_lin) + '\n')

from sklearn.linear_model import Ridge

ridgereg = Ridge()
ridgereg.fit(X_train,y_train)
y_pred = ridgereg.predict(X_test)

error_reg = np.sqrt(metrics.mean_absolute_error(y_test,y_predict))
print (error_reg)
f.write("Ridge Regression" + ',' + str(error_reg) + '\n')

from sklearn import ensemble

model = ensemble.ExtraTreesRegressor(n_jobs=5, random_state=0)
reg.fit(X_train, y_train)

y_predict = reg.predict(X_test)

error_tree = np.sqrt(metrics.mean_absolute_error(y_test,y_predict))
print (error_tree)
f.write("Extra Trees Regressor" + ',' + str(error_tree) + '\n')

import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# The Input Layer :
model.add(Dense(64, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
model.add(Dense(128, kernel_initializer='normal',activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(64, kernel_initializer='normal',activation='relu'))

# The Output Layer :
model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()

model.fit(X_train,y_train,validation_split=0.1,batch_size=1024,epochs=1000)

y_predict = model.predict(X_test)

error_neural = np.sqrt(metrics.mean_absolute_error(y_test,y_predict))
f.write("Deep Neural Network" + ',' + str(error_neural) + '\n')


plt.title("Predited Data Vs Test Data")
plt.ylim(1.6, 2.2)
plt.plot(pd.Series(np.arange(len(y_test))),np.log10(y_test),color = 'yellow', linewidth=0.25, linestyle="-" )
plt.plot(pd.Series(np.arange(len(y_predict))),np.log10(y_predict),color = 'red', linewidth=1, linestyle="-" )
plt.show()








