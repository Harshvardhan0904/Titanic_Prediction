#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder , OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score , train_test_split , StratifiedKFold ,GridSearchCV



# In[2]:


train = pd.read_csv(r"C:/Users/harsh/Desktop/data Analysis/train.csv")
test = pd.read_csv(r"C:/Users/harsh/Desktop/data Analysis/test.csv")


# In[3]:


train.head()


# In[4]:


train.info()


# In[5]:


train.describe()


# In[6]:


train.groupby(['Pclass'],as_index=False)['Survived'].mean()


# In[7]:


train.groupby(['Sex'],as_index=False)['Survived'].mean()


# In[8]:


train.groupby(['SibSp'],as_index=False)['Survived'].mean()


# In[9]:


train.groupby(['Parch'],as_index=False)['Survived'].mean()


# In[10]:


train["Family_size"] = train["SibSp"]+train['Parch']+1
test["Family_size"] = train["SibSp"]+train['Parch']+1
train.head(10)


# In[11]:


train.groupby(["Family_size"], as_index=False)["Survived"].mean()


# In[12]:


family_map = {1:"Alone" , 2:"small",3:"small",4:"small" , 5:"medium" , 6:"medium" , 7:"Large", 8:"Large"}
train['Family_map'] = train['Family_size'].map(family_map)
test['Family_map'] = train['Family_size'].map(family_map)
train.head()


# In[13]:


sns.displot(train, x='Age' , col="Survived" , binwidth = 10 , height =5)


# In[14]:


train['Age_cut']=pd.qcut(train['Age'],8)
test['Age_cut']=pd.qcut(test['Age'],8)


# In[15]:


train.groupby(['Age_cut'],as_index=True)['Survived'].mean()


# In[16]:


# For the train dataset
train.loc[train['Age'] <= 16, "Age"] = 0
train.loc[(train["Age"] > 16) & (train["Age"] <= 20), "Age"] = 1
train.loc[(train["Age"] > 20) & (train["Age"] <= 24), "Age"] = 2
train.loc[(train["Age"] > 24) & (train["Age"] <= 28), "Age"] = 3
train.loc[(train["Age"] > 28) & (train["Age"] <= 32), "Age"] = 4
train.loc[(train["Age"] > 32) & (train["Age"] <= 38), "Age"] = 5
train.loc[(train["Age"] > 38) & (train["Age"] <= 47), "Age"] = 6
train.loc[(train["Age"] > 47) & (train["Age"] <= 80), "Age"] = 7
train.loc[train["Age"] > 80, "Age"] = 8

# For the test dataset
test.loc[test['Age'] <= 16, "Age"] = 0
test.loc[(test["Age"] > 16) & (test["Age"] <= 20), "Age"] = 1
test.loc[(test["Age"] > 20) & (test["Age"] <= 24), "Age"] = 2
test.loc[(test["Age"] > 24) & (test["Age"] <= 28), "Age"] = 3
test.loc[(test["Age"] > 28) & (test["Age"] <= 32), "Age"] = 4
test.loc[(test["Age"] > 32) & (test["Age"] <= 38), "Age"] = 5
test.loc[(test["Age"] > 38) & (test["Age"] <= 47), "Age"] = 6
test.loc[(test["Age"] > 47) & (test["Age"] <= 80), "Age"] = 7
test.loc[test["Age"] > 80, "Age"] = 8


# In[17]:


train.head()


# In[18]:


train['Fare_cut']=pd.qcut(train['Fare'],5)
test['Fare_cut']=pd.qcut(test['Fare'],5)


# In[19]:


train.groupby(['Fare_cut'],as_index=True)['Survived'].mean()


# In[20]:


train.loc[train['Fare'] <= 7, "Fare"] = 0
train.loc[(train["Fare"] > 7) & (train["Fare"] <= 10), "Fare"] = 1
train.loc[(train["Fare"] > 10) & (train["Fare"] <= 21), "Fare"] = 2
train.loc[(train["Fare"] > 21) & (train["Fare"] <= 39), "Fare"] = 3
train.loc[(train["Fare"] > 39) & (train["Fare"] <= 512), "Fare"] = 4
train.loc[train["Fare"] > 512, "Fare"] = 5 

test.loc[test['Fare'] <= 7, "Fare"] = 0
test.loc[(test["Fare"] > 7) & (test["Fare"] <= 10), "Fare"] = 1
test.loc[(test["Fare"] > 10) & (test["Fare"] <= 21), "Fare"] = 2
test.loc[(test["Fare"] > 21) & (test["Fare"] <= 39), "Fare"] = 3
test.loc[(test["Fare"] > 39) & (test["Fare"] <= 512), "Fare"] = 4
test.loc[test["Fare"] > 512, "Fare"] = 5


# In[21]:


train["Name"]


# In[22]:


split = train["Name"].str.split(pat = ",",expand = True) #expand = fasle mean it will print the result into list form
split.columns = ["First Name" , "Second Name"]
split


# In[23]:


train['Title'] = train["Name"].str.split(pat = ",",expand = True)[1].str.split(pat = "." , expand = True)[0].apply(lambda x: x.strip())

#apply(lambda x: x.strip()) used for removing trailing and leading spaces 

test['Title'] = test["Name"].str.split(pat = ",",expand = True)[1].str.split(pat = "." , expand = True)[0].apply(lambda x: x.strip())


# In[24]:


train.groupby(['Title'],as_index=True)['Survived'].mean()


# In[25]:


train["Title"]=train["Title"].replace({
    "Capt":"Militray",
     "Col":"Militray",
    "Major":"Militray",
    "Jonkheer":"Noble",
    "the Countess":"Noble",
    "Don":"Noble",
    "Lady":"Noble",
    "Sir":"Noble",
    "Mme":"Noble",
    "Mlle":"Noble",
    "Ms":"Noble"
})
test["Title"]=test["Title"].replace({
    "Capt":"Militray",
     "Col":"Militray",
    "Major":"Militray",
    "Jonkheer":"Noble",
    "the Countess":"Noble",
    "Don":"Noble",
    "Lady":"Noble",
    "Sir":"Noble",
    "Mme":"Noble",
    "Mlle":"Noble",
    "Ms":"Noble"
})


# In[26]:


train.head(10)


# In[27]:


train.groupby(['Title'],as_index=True)['Survived'].agg(['count' , 'mean'])


# In[28]:


train["Cabin"] = train["Cabin"].fillna("U")
test["Cabin"] = test["Cabin"].fillna("U")


# In[29]:


train["Cabin"]


# In[30]:


train.groupby(['Cabin'],as_index = False)["Survived"].agg(['mean','count'])
train["newCabin"]=train['Cabin'].str.extract(r'([A-Za-z]+)(\d*)')[0]
test["newCabin"]=test['Cabin'].str.extract(r'([A-Za-z]+)(\d*)')[0]


# In[31]:


train['Cabin_assinged']=train["newCabin"].apply(lambda x : 0 if x in ['U'] else 1)
test['Cabin_assinged']=test["newCabin"].apply(lambda x : 0 if x in ['U'] else 1)


# In[32]:


train.groupby(['Cabin_assinged'],as_index = False)["Survived"].agg(['count','mean'])


# In[33]:


test.shape


# In[34]:


train.info()


# In[35]:


train["Age"] = train["Age"].fillna(train["Age"].mean())
test["Age"] = test["Age"].fillna(test["Age"].mean())
test["Fare"] = test["Fare"].fillna(test["Fare"].mean())


# In[36]:


ohe = OneHotEncoder(sparse_output= False)
ode = OrdinalEncoder
SI = SimpleImputer(strategy = 'most_frequent')


# In[38]:


ohe_col = ["Family_map", "newCabin"]  # Columns to be one-hot encoded
ode_col = ['Sex', 'Embarked', 'Title']


# In[39]:


X = train.drop(["Survived", "Ticket", "Cabin", "Name"], axis=1)
y = train["Survived"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=21)


# In[40]:


ode_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
ohe_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


# In[41]:


from sklearn.compose import ColumnTransformer


# In[42]:


simple_processor = ColumnTransformer(transformers=[
    ('ohe_pipeline', ohe_pipeline, ohe_col),  # OneHotEncode specific columns
    ('ode_pipeline', ode_pipeline, ode_col),  # OrdinalEncode specific columns
    ('passthrough', 'passthrough', ['Pclass', 'Fare'])  # Pass numeric columns directly
])

simple_pipeline = make_pipeline(simple_processor, RandomForestClassifier())
simple_pipeline.fit(X_train, y_train)



# In[44]:


rf= RandomForestClassifier()
param_grid = {
    'n_estimators' : [100, 150, 200],
    'min_samples_split': [5, 10, 15],
    'max_depth' : [8, 9, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4],
    'criterion' : ['gini', 'entropy']
}


# In[45]:


# Simplified pipeline to test
processor = ColumnTransformer(transformers=[
    ('ohe_pipeline', ohe_pipeline, ohe_col),  # OneHotEncode specific columns
    ('ode_pipeline', ode_pipeline, ode_col),  # OrdinalEncode specific columns
    ('passthrough', 'passthrough', ['Pclass', 'Fare'])  # Pass numeric columns directly
])

# Data after preprocessing
X_preprocessed = processor.fit_transform(X_train)
print(pd.DataFrame(X_preprocessed).head())

# Check data types before fitting the model
print(X_train.dtypes)

# Simple RandomForestClassifier to check if issue persists without GridSearch
rf_simple = RandomForestClassifier()
rf_simple.fit(X_train.select_dtypes(include=[np.number]), y_train)

# If the above works, try fitting the pipeline
simple_pipeline = make_pipeline(processor, RandomForestClassifier())
simple_pipeline.fit(X_train, y_train)


# In[46]:


CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=StratifiedKFold(n_splits=5), error_score='raise')


# In[47]:


# Simplified pipeline to test
processor = ColumnTransformer(transformers=[
    ('ohe_pipeline', ohe_pipeline, ohe_col),  # OneHotEncode specific columns
    ('ode_pipeline', ode_pipeline, ode_col),  # OrdinalEncode specific columns
    ('passthrough', 'passthrough', ['Pclass', 'Fare'])  # Pass numeric columns directly
])

# Data after preprocessing
X_preprocessed = processor.fit_transform(X_train)
print(pd.DataFrame(X_preprocessed).head())

print(X_train.dtypes)

rf_simple = RandomForestClassifier()
rf_simple.fit(X_train.select_dtypes(include=[np.number]), y_train)

simple_pipeline = make_pipeline(processor, RandomForestClassifier())
simple_pipeline.fit(X_train, y_train)


# In[48]:


pipeFinal = make_pipeline(processor,CV_rf)
pipeFinal.fit(X_train,y_train)


# In[49]:


print(CV_rf.best_params_)
print(CV_rf.best_score_*100)


# In[50]:


dc =DecisionTreeClassifier()
param_grid = {
    'min_samples_split': [5, 10, 15],
    'max_depth' : [10, 20,30],
    'min_samples_leaf': [1, 2, 4],
    'criterion' : ['gini', 'entropy']
}
CV_dc = GridSearchCV(estimator=dc, param_grid=param_grid, cv=StratifiedKFold(n_splits=5), error_score='raise')


# In[51]:


pipeFinal = make_pipeline(processor,CV_dc)
pipeFinal.fit(X_train,y_train)

print(CV_dc.best_params_)
print(CV_dc.best_score_*100)


# In[ ]:




