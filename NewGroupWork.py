#!/usr/bin/env python
# coding: utf-8

# In[1]:


# download all the libraries required for this group assignment
import numpy as np
import pandas as pd
import io
import requests
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
import os
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load both the normal and testing data
ReadAdultData = pd.read_excel('data/adult.data.xlsx')
# ReadAdultTestData = pd.read_excel('adult.test.xlsx')


# In[3]:


# show the normal data
ReadAdultData.head()


# In[4]:


# get information from the ReadAdultData
print('Information from the ReadAdultData\n')
ReadAdultData.info()
# get information from the ReadAdultTestData
# print('\n\n\nInformation from the ReadAdultTestData\n')
# ReadAdultTestData.info()


# <h2>Desceribe the Dataset</h2>

# In[5]:


ReadAdultData.describe(include='all')


# <h2>Check for missing values</h2>

# In[6]:


ReadAdultData.isnull().sum()


# In[7]:


# check for the shape of the variable
ReadAdultData.shape


# <h2>Rename Column</h2>

# In[8]:


ReadAdultData.rename(columns={'Work Class':'Work_Class', "Final Weight":'Final_Weight', "Education Number of Years":'Education_Number_of_Years', "Marital Status":'Marital_Status', "Capital Gain":'Capital_Gain', "Capital Loss":'Capital_Loss', "Hours per Week":'Hours_per_Week', "Native Country":'Native_Country'}, inplace=True)


# <h1>To many income classes</h1><br>
# First of all, we can see that the income_class column has 4 unique classes, but we expect to have only two.

# In[9]:


ReadAdultData.Income.value_counts(dropna=False)


# <h1>Extra category</h1><br>
# If we compare number of unique categorical features for other variables it's easy to see that workclass, occupation and native_country have one extra unique value (+1 unique values compare to the description from the Dataset Description.odt) in the data. One extra class equals to ?. It looks like this values has to be replaced with NaN.

# In[10]:


ReadAdultData.Occupation.value_counts(dropna=False).to_frame()


# In[11]:


ReadAdultData.Work_Class.value_counts(dropna=False).to_frame()


# <p>We can confirm that those are the only columns that have value equal to ?</p>

# In[12]:


(ReadAdultData == ' ?').sum(axis=0)


# <h1>Suspicios values with 9s</h1>
# Two columns have max value with all 9s in them. It's common that people use values with all 9s in order to mark NaN values in columns with numerical values.

# In[13]:


ReadAdultData.Capital_Gain.value_counts(dropna=False).head(10)


# In[14]:


import heapq
heapq.nlargest(10, ReadAdultData.Capital_Gain.unique())


# In[15]:


import heapq
heapq.nlargest(10, ReadAdultData.Hours_per_Week.unique())


# In case of hours_per_week we can see that there are some unique values that close to 99

# In[16]:


Hours_per_Week_Counts = ReadAdultData.Hours_per_Week.value_counts()
largest_hours_per_week = heapq.nlargest(10, ReadAdultData.Hours_per_Week.unique())
Hours_per_Week_Counts[Hours_per_Week_Counts.index.isin(largest_hours_per_week)]


# If we check countwise it's clear that 99 is unusually large compare to other values. It also could be that 99 means 99+ hours and long tail might fit into this category.

# <h1>Check for Duplicates</h1>

# In[17]:


n_samples_before = ReadAdultData.shape[0]
n_samples_after = ReadAdultData.drop_duplicates().shape[0]

print(n_samples_before)
print(n_samples_after)
print("Duplicates removed: {:.2%}".format((n_samples_before - n_samples_after) / n_samples_before))


# <h1>Apply first cleaning steps</h1><br>
# These steps allow us to address biggest issues that we found so far. Without this fixes it will be harder to do data analysis.

# After removing final weight column we get 10% of duplicates in the training data. We need to remove it before continuing our analysis in order to avoid biases.

# In[18]:


def clean_dataset( ReadAdultData):
    # Test dataset has dot at the end, we remove it in order
    # to unify names between training and test datasets.
    ReadAdultData['Income'] = ReadAdultData.Income.str.rstrip('.').astype('category')
    
    # Remove final weight column since there is no use
    # for it during the classification.
    ReadAdultData = ReadAdultData.drop('final_weight', axis=1)
    
    # Duplicates might create biases during the analysis and
    # during prediction stage they might give over-optimistic
    # (or pessimistic) results.
    ReadAdultData = ReadAdultData.drop_duplicates()

    return ReadAdultData


# <h1> Visualization for the columns</h1>

# <h3>Work Class column </h3>

# In[19]:


sns.countplot(x = ReadAdultData['Work_Class'])
plt.xticks(rotation = 90)


# <h1>Education COlumn</h1>

# In[20]:


sns.countplot(x = ReadAdultData['Education'])
plt.xticks(rotation = 90)


# <h3>Marital Status Column</h3>

# In[21]:


sns.countplot(x = ReadAdultData['Marital_Status'])
plt.xticks(rotation = 90)


# <h3>Occupation Column</h3>

# In[22]:


sns.countplot(x = ReadAdultData['Occupation'])
plt.xticks(rotation = 90)


# <h3>Relationship Column</h3>

# In[23]:


sns.countplot(x = ReadAdultData['Relationship'])
plt.xticks(rotation = 90)


# <h3>Race Column</h3>

# In[24]:


sns.countplot(x = ReadAdultData['Race'])
plt.xticks(rotation = 90)


# <h3>Gender Column</h3>

# In[25]:


sns.countplot(x = ReadAdultData['Sex'])
plt.xticks(rotation = 90)


# <h2>Captital Gain and Loss against Genders(sex)</h2>

# In[26]:


sns.barplot(x = ReadAdultData['Capital_Gain'], y= ReadAdultData['Sex'])
plt.xticks(rotation = 90)


# In[27]:


sns.barplot(x = ReadAdultData['Capital_Loss'], y= ReadAdultData['Sex'])
plt.xticks(rotation = 90)


# <h3>Native Country Column</h3>

# In[28]:


sns.countplot(x = ReadAdultData['Native_Country'])
plt.xticks(rotation = 90)


# <h3>Income Column</h3>

# In[29]:


sns.countplot(x = ReadAdultData['Income'])
plt.xticks(rotation = 90)


# <h3>Income and Race Column</h3>

# In[30]:


sns.countplot(x='Income', hue='Race', data = ReadAdultData)


# <h3>Income and Gender Column</h3>

# In[31]:


sns.countplot(x='Income', hue='Sex', data = ReadAdultData)


# <h3>Income and Workclass Column</h3>

# In[32]:


sns.countplot(x='Work_Class', hue='Income', data = ReadAdultData)
plt.xticks(rotation = 90)


# <h3>Income and Education Column</h3>

# In[33]:


sns.countplot(x='Education', hue='Income', data = ReadAdultData)
plt.xticks(rotation = 90)


# <h3>Income and Relationship Column</h3>

# In[34]:


sns.countplot(x='Relationship', hue='Income', data = ReadAdultData)
plt.xticks(rotation = 90)


# <h3>Income and Occupation Column</h3>

# In[35]:


sns.countplot(x='Occupation', hue='Income', data = ReadAdultData)
plt.xticks(rotation = 90)


# <h3>Columns that Contains Numerical Datasets</h3>

# In[36]:


ReadAdultData.hist(figsize=(10,10))


# <h3>Correlation between Numerical columns</h3>
# <p style="font-size:15px">Income has 34% correlation with ‘Education_num’, 23% correlation with ‘hours_per_week’ and ‘age’, and 22% correlation with ‘Capital_gain’. The correlations are moderate.</p>

# In[37]:


sns.heatmap(ReadAdultData.corr(), annot=True)


# <h2> Capture the columns with missing values and work on them using mode</h2>

# In[38]:


missingVualuesColumn = ['Work_Class', 'Occupation', 'Native_Country']
for col in missingVualuesColumn:
      ReadAdultData[col].fillna(ReadAdultData[col].mode()[0], inplace=True)


# <h2> Conversion of categorical to numerical</h2>

# In[39]:


from sklearn.preprocessing import LabelEncoder
ConvertCat =  LabelEncoder()
ReadAdultData.columns


# <h2>Select Categorical Columns to Encode</h2>

# <h2>Check for the uniqueness of the income column and convert it to 0's ad 1's</h2>

# In[40]:


cat_col_to_encode= ['Work_Class','Education','Marital_Status', 'Occupation', 'Relationship', 'Race','Sex','Native_Country']


# In[41]:


print(pd.unique(ReadAdultData['Income']))


# In[42]:


def income(options):
    if options == ' <=50K':
        return 0
    if options == ' >50K':
        return 1


# In[43]:


ReadAdultData['Income'] = ReadAdultData['Income'].apply(income)


# In[44]:


ReadAdultData['Income']


# In[45]:


ReadAdultData_new = pd.get_dummies(ReadAdultData, columns=cat_col_to_encode, drop_first=True)


# In[46]:


ReadAdultData_new.head()


# <h2>Normalization of DataSet</h2>
# <p>The next step is to normalize the data, since there are certain columns with very small values and some columns with high values. This process is important as values on a similar scale allow the model to learn better.
# We use standard scaler for this process –
# ‘StandardScaler follows Standard Normal Distribution (SND). Therefore, it makes mean = 0 and scales the data to unit variance’</p>

# In[47]:


from sklearn.preprocessing import StandardScaler


# In[48]:


ScalerData = StandardScaler()


# In[49]:


ScalingData = ReadAdultData_new.drop('Income', axis='columns')
dataY = ReadAdultData_new['Income']
print(dataY)


# In[50]:


ScaledData = ScalerData.fit_transform(ScalingData)
dataX = pd.DataFrame(ScaledData, columns=ScalingData.columns)
print(dataX)


# In[51]:


training_data = pd.concat([dataY, dataX], axis=1,join='inner')
training_data.head()


# <h2>Feature Selection</h2>

# In[52]:


from sklearn.feature_selection import SelectKBest,chi2
np.seterr(divide='ignore',invalid='ignore')
Feature_selector=SelectKBest(k=training_data.shape[1])


# In[53]:


training_selected_features=Feature_selector.fit_transform(training_data, dataY)
selected_cols = Feature_selector.get_support(indices=True)

# selected features
selected_feature_names = training_data.columns.values[selected_cols]
training_selected_features = pd.DataFrame(training_selected_features)


# In[54]:


scores = Feature_selector.scores_[Feature_selector.get_support()]
selected_feature_names_scores = list(zip(selected_feature_names, scores))


# In[55]:


Feat_F1score_combined = pd.DataFrame(data = selected_feature_names_scores, columns=['Feature_names', 'F_Scores'])
Feat_F1score_combined = Feat_F1score_combined.sort_values(['F_Scores', 'Feature_names'], ascending = [False, True])


# In[56]:


Feat_F1score_combined.plot(x='Feature_names',y='F_Scores',kind='bar',title='Fscores of features arranged in accordance with their importance using SelectKBest method',figsize=(18,8))
#Setting the F score threshold as 30, we get a total of 30 features which have F scores beyond this value

kbest_selector=SelectKBest(k=15)
training_selected_features=kbest_selector.fit_transform(training_data,dataY)


# <h2>Models</h2>

# In[57]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[58]:


DTC = DecisionTreeClassifier()
GNB = GaussianNB()
SVC = SVC()
KNN = KNeighborsClassifier()
LG = LogisticRegression()


# In[59]:


x_train, x_test, y_train, y_test = train_test_split(training_data,dataY, test_size = 0.2, random_state = 101)
for i in [DTC, GNB, SVC, KNN, LG]:
    i.fit(x_train, y_train)
    prediction = i.predict(x_test)
    test_score = accuracy_score(y_test, prediction)
    train_score = accuracy_score(y_train, i.predict(x_train))
    if abs(train_score - test_score) <= 0.01:
        print(i)
        print('Accuracy score for train data: ' , accuracy_score(y_test, prediction))
        print('Accuracy score for test data: ' , accuracy_score(y_train, i.predict(x_train)))
        print(classification_report(y_test, prediction))
        print(confusion_matrix(y_test, prediction))
        print('\n--------------------------------------------\n')


# In[60]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
RFC = RandomForestClassifier()
AD = AdaBoostClassifier()
GD = GradientBoostingClassifier()


# In[61]:


x_train, x_test, y_train, y_test = train_test_split(training_data, dataY, test_size = 0.2, random_state = 101)
for i in [RFC, AD, GD]:
    i.fit(x_train, y_train)
    prediction = i.predict(x_test)
    test_score = accuracy_score(y_test, prediction)
    train_score = accuracy_score(y_train, i.predict(x_train))
    if abs(train_score - test_score) <= 0.01:
        print(i)
        print('Accuracy score for train data: ' , accuracy_score(y_test, prediction))
        print('Accuracy score for test data: ' , accuracy_score(y_train, i.predict(x_train)))
        print(classification_report(y_test, prediction))
        print(confusion_matrix(y_test, prediction))
        print('\n--------------------------------------------\n')


# <h3>Cross Validation</h3>
# <p>The goal of cross-validation is to test the model’s ability to predict new data that was not used in estimating it, in order to flag problems like overfitting or selection bias and to give an insight on how the model will generalize to an independent dataset (i.e., an unknown dataset, for instance from a real problem)</p>

# In[62]:


from sklearn.model_selection import cross_val_score
for i in range(2,10):
    cv = cross_val_score(GD, training_data, dataY, cv = i)
    print(GD, cv.mean())


# In[63]:


from sklearn.model_selection import cross_val_score
for i in range(2,10):
    cv = cross_val_score(RFC, training_data, dataY, cv = i)
    print(RFC, cv.mean())


# In[64]:


from sklearn.model_selection import cross_val_score
for i in range(2,100):
    cv = cross_val_score(RFC, training_data, dataY, cv = 11)
    print(RFC, cv.mean())


# In[ ]:





# In[ ]:




