# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

# CODE AND OUPUT:
```
NAME : R . JOYCE BEULAH
REG NO : 212222230058
```


```
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2

df=pd.read_csv("/content/titanic_dataset.csv")

df.columns
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/f12b92dd-2b99-4112-b503-f6621e188376)
```
df.shape
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/8bff8a64-a204-406d-b94c-d79a2d9d11f0)
```
x=df.drop("Survived",1)
y=df['Survived']
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/3d3995cb-3574-48e1-98fd-abb65decf8c4)

```
df1=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)
df1.columns
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/777a3891-8d67-4730-b88f-a1b302559dc5)

```
df1['Age'].isnull().sum()
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/941c441a-571b-4f50-a2ca-609aca3035df)

```
df1['Age'].fillna(method='ffill')
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/a57b5018-113f-4b6d-ad67-a8cf4036b400)

```
df1['Age']=df1['Age'].fillna(method='ffill')
df1['Age'].isnull().sum()
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/0384839c-ffd0-455f-8c0d-366a5877592e)

```
feature=SelectKBest(mutual_info_classif,k=3)
df1.columns
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/8d83bcfe-7ef0-4977-a254-8ef27aad0faf)

```
cols=df1.columns.tolist()
cols[-1],cols[1]=cols[1],cols[-1]
df1.columns
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/8b7b083c-a7f1-48f0-ae16-882b0b463814)

```
x=df1.iloc[:,0:6]
y=df1.iloc[:,6]

x.columns
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/13d45295-4f3b-4186-8dad-40f25412f2f6)

```
y=y.to_frame()
y.columns
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/dda29eb8-f135-483c-acc2-2551e996326c)

```
from sklearn.feature_selection import SelectKBest

data=pd.read_csv("/content/titanic_dataset.csv")
data=data.dropna()
x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']
x
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/5104376f-476a-4307-92a1-a17b3ad8a502)

```
data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data[ "Embarked" ]=data ["Embarked"] .astype ("category")

data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data[ "Embarked" ]=data ["Embarked"] .cat.codes
data
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/7c4b2301-0b65-4474-a5a7-3d52cc0b82c6)

```
k=5
selector = SelectKBest(score_func=chi2,k=k)
x_new = selector.fit_transform(x,y)

selected_feature_indices = selector.get_support(indices=True)

selected_feature_indices = selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features: ")
print(selected_features)
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/ed5377b2-0d14-4e6b-a9d7-6f4f158ba179)

```
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
sfm = SelectFromModel(model, threshold='mean')
sfm.fit(x,y)
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/e09fe730-1ead-4544-8b5e-55eeb1389fb0)

```
selected_feature = x.columns[sfm.get_support()]

print("Selected Features:")
print(selected_feature)
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/68a7feb8-53e3-4e98-9173-7e091cee5233)

```
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression()
num_features_to_remove =2
rfe = RFE(model, n_features_to_select=(len(x.columns) - num_features_to_remove))
rfe.fit(x,y)
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/510b30f9-2b30-4893-ac51-5e24a054d5fd)

```
selected_features = x.columns[rfe.support_]
print("Selected Features:")
print(selected_feature)
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/65ba56f6-29ee-4a05-bdbf-8107b905fb09)

```
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x,y)
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/40660088-e662-4b1b-9080-1c0aaadcfb27)

```
feature_importances = model.feature_importances_
threshold = 0.15
selected_features = x.columns[feature_importances > threshold]

print("Selected Features:")
print(selected_feature)
```
![image](https://github.com/JoyceBeulah/ODD2023-Datascience-Ex-07/assets/118343698/fbc287cf-44b1-404c-96b2-eb3b162649a5)


# RESULT :
Thus, the various feature selection techniques have been performed on a given dataset successfully.
