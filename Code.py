#Importing libraries
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, linear_model, metrics 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.feature_selection import SelectKBest,RFE,RFECV
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np

#KNOWING DATA
#id - booking ID 
#user_id - the ID of the customer (based on mobile number)
#vehicle_model_id - vehicle model type.
#package_id - type of package (1=4hrs & 40kms, 2=8hrs & 80kms, 3=6hrs & 60kms, 4= 10hrs & 100kms, 5=5hrs & 50kms, 6=3hrs & 30kms, 7=12hrs & 120kms)
#travel_type_id - type of travel (1=long distance, 2= point to point, 3= hourly rental).
#from_area_id - unique identifier of area. Applicable only for point-to-point travel and packages
#to_area_id - unique identifier of area. Applicable only for point-to-point travel
#from_city_id - unique identifier of city
#to_city_id - unique identifier of city (only for intercity)
#from_date - time stamp of requested trip start
#to_date - time stamp of trip end
#online_booking - if booking was done on desktop website
#mobile_site_booking - if booking was done on mobile website
#booking_created - time stamp of booking
#from_lat - latitude of from area
#from_long -  longitude of from area
#to_lat - latitude of to area
#to_long - longitude of to area
#Car_Cancellation (available only in training data) - whether the booking was cancelled (1) or not (0) due to unavailability of a car.
#Cost_of_error (available only in training data) - the cost incurred if the booking is misclassified. For an un-cancelled booking, the cost of misclassification is 1. For a cancelled booking, the cost is a function of the cancellation time relative to the trip start time

cabs_training = pd.read_csv('Kaggle_YourCabs_training.csv')

#As you can see there are various columns with missing values
#Date related columns should be converted to date-time datatype instead of float.

cabs_training.shape


#data types of each column in csv file
print(cabs_training.dtypes)


#Getting unique entries for each column
#for col in cabs_training:
#   print(col,cabs_training[col].unique(),'\n')

# Converting required columns to date-time data type
# Note: when we try to convert NaN to date-time, it gets converted to NaT( Not a timestamp )
#date_column is given non date time format so has been converted to date time format
#to_date column,booking_created column,from_date
cabs_training['from_date'] = pd.to_datetime(cabs_training['from_date'])
cabs_training['to_date'] = pd.to_datetime(cabs_training['to_date'])
cabs_training['booking_created'] = pd.to_datetime(cabs_training['booking_created'])

#count shows that there are several columns which have missing values like package_id ,to_area_is,from_city_id etc(actual total is 4341)
#Handling missing values before classifying
cabs_training.describe()

#id is set as index
#inplace true return no object and does a changes in the data itself
#inplace false return an object after doing the changes  and original data is not impacted
cabs_training.set_index('id', inplace=True)


# In[9]:


#index column is not shown 
#Datetime datatype is reflected
cabs_training.info()


# In[10]:


# Finding out if class imbalance is present or not
#There is huge class imbalance problem here so chances of getting high accuracy without giving accurate result for 1 class is high
#The same is dealt before classifying the data
cabs_training.Car_Cancellation.value_counts()


# In[12]:


#autopct is just format specifier
plt.pie(cabs_training.Car_Cancellation.value_counts(),labels=['Class 0','Class 1'],autopct='%1.2f%%')
plt.show()


# In[ ]:


#BEGIN VISULIZATION PART


# In[34]:


#most people opt for package that is 4hrs & 40kms followed by 8hrs & 80kms.
cabs_training.package_id.value_counts()
plt.hist(cabs_training.package_id,color='green',bins=15)
plt.title('Frequency vs Package Type ')
plt.ylabel('Frequency')
plt.xlabel('package_id')
plt.show()

#the entire data set is divides into n number of equal parts from maximum to minimum value called bins.Each datpoint reside in only one bin
#shows most of the cancellations are in actully in the package type which is most opted for i.e 4hrs & 40kms  followed by 8hrs & 80kms and 3hrs & 30kms
cabs_training.hist(column='package_id', by='Car_Cancellation',bins=7,color='green');
#Analysis: Shorter distance routes are more preferred.Might be because the areas are close to people hometown or because areas are nearby hill station  


# In[32]:


#distribution for travel_type 
cabs_training.travel_type_id.value_counts()
#most people opt for point to point travelling.
plt.hist(cabs_training.travel_type_id,color='green')
plt.title('Frequency vs Travel Type ID ')
plt.xticks(range(1,4))
plt.ylabel('Frequency')
plt.xlabel('travel_type_id')
plt.show()

#we can observe that that point to point travelling type is the most cancelled
#Analysis: Same as in package ID
cabs_training.hist(column='travel_type_id', by='Car_Cancellation',color='green');


# In[39]:


#most people prefer booking offline rather than through mobile/online services.
plt.hist([cabs_training.online_booking,cabs_training.mobile_site_booking], bins=10, label=['online_booking', 'mobile_site_booking'])
plt.xticks(range(0,2))
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()

#If any online booking has been done the number of cancellations done are comparitively more than when done offline
online= cabs_training.groupby(['online_booking','Car_Cancellation']).size().unstack(1)
online.head()
online.plot(kind='bar')

#If any online booking has been done the number of cancellations done are comparitively less than when done offline
mobile= cabs_training.groupby(['mobile_site_booking','Car_Cancellation']).size().unstack(1)
mobile.head()
mobile.plot(kind='bar')

# To check how online/mobile site booking is impacting cab cancellations
offline=pd.pivot_table(cabs_training[['online_booking', 'mobile_site_booking', 'Car_Cancellation']]
              , values='Car_Cancellation', index=['online_booking', 'mobile_site_booking'], aggfunc= np.count_nonzero)
offline.head()

#It is observed that the number of cancellations are greater if the bookings have been made online and not offline
offline.plot(kind='bar',color='green')

#Analysis: People prefer doing online booking as compared to offilne booking as it saves time and money and less hectic as weel
#          Out of the two online is more famous than mobile booking as less number of people would be having access to the mobile phone
#          but a lot of people would be having internet cafes nearby to acces internet and do online booking
#          or maybe not many people have app installed if that what is considered as mobile booking


# In[45]:


# Lets take a look at from_area_id variable and see if it has any relation with car cancellation.
from_area= cabs_training.groupby(['from_area_id', 'Car_Cancellation']).size().unstack(1)
from_area.head()

from_area['percent_cancelled'] = (from_area[1] / (from_area[1] + from_area[0])) * 100.

#If the starting location is area of id 130 then the cabs are mostly cancelled.
from_area.percent_cancelled.sort_values(ascending=False).iloc[:10]

#Analysis: That region might be far or the cab drivers do not get a return customer so a loss on their part
#          or the region might not be a good place interms of environment , roads, refreshments, goons presence etc


# In[46]:


to_area = cabs_training.groupby(['to_area_id', 'Car_Cancellation']).size().unstack(1)
to_area['percent_cancelled'] = (to_area[1] / (to_area[1] + to_area[0])) * 100.
#If the destination is to area id 1247 then the chances that ride would be cancelled is more
to_area.percent_cancelled.sort_values(ascending=False).iloc[:10]

#Analysis: same as from area id


# In[49]:


#Analysis: same as from area id
to_fro = cabs_training.groupby(['from_area_id', 'to_area_id', 'Car_Cancellation']).size()
f_t = to_fro.unstack(2)
f_t.head()
f_t['percent_cancelled'] = (f_t[1] / (f_t[0] + f_t[1])) * 100.
f_t.percent_cancelled.sort_values(ascending=False).iloc[:20]


# In[50]:


#trying to visualize if booking and travelling date difference has any impact on Car Cancellation
booking_diff = cabs_training[['from_date', 'to_date', 'booking_created', 'Car_Cancellation']]
booking_diff.loc[:, 'difference'] = booking_diff.loc[:, ('from_date')] - booking_diff.loc[:, ('booking_created')]
from datetime import datetime
def convert_to_days(x):
    days = round(x.seconds / ( 24 * 60 * 60 ),5)
    return days
booking_diff.loc[:, 'diff_in_days'] = booking_diff.difference.map(convert_to_days)

#the journey day and the booking day is at max 1 day differnce.Hence does not seem any realtion between the day of booking and the daya of travel
#As in we cannot guess if as the time difference between the booking and travelling increase or decrease any change in the booking being cancelled is there
booking_diff=booking_diff.sort_values(by=['diff_in_days'],ascending=False)
booking_diff.diff_in_days

#not providing much useful information hence we will see in a more narrower sence.
booking_diff.hist(column='diff_in_days', by='Car_Cancellation', bins=200);

#Analysis: as mostly bookings are point to point so chances are less that people will book quite a no of days earlier before their travel day


# In[266]:


#vehicle id 12 is mostly used for the cab service
cabs_training.vehicle_model_id.value_counts()
#Analysis: the Yourcabs service might have more car model of that kind as it might have cut a deal with such car making model company


# In[267]:


cabs_training.loc[:, 'from_month'] = cabs_training.from_date.dt.month
cabs_training.loc[:, 'from_weekday'] = cabs_training.from_date.dt.weekday
cabs_training.loc[:, 'booking_month'] = cabs_training.booking_created.dt.month
cabs_training.loc[:, 'booking_weekday'] = cabs_training.booking_created.dt.weekday

#August month have seen the highest number of bookings followed by september, october
#Analysis :These are months are generally festivals  month  of rakhshanbhan coming up with diwali so poeple 
#          be going to their hometown or rainy season always attracts a visit to nearby hill station or a road trip
plt.hist(cabs_training.from_month,color='green',bins=30)
plt.xticks(range(1,13))
plt.ylabel('Frequency')
plt.xlabel('month')
plt.show()

month= cabs_training.groupby(['from_month', 'Car_Cancellation']).size().unstack(1)
month.head()
#booking in the month of October are majorly cancelled
month.plot(kind='bar')

#weekends and friday has the max booking
#weekends and fridays suffer max cancellation
#Analysis: Demand od the cabs are likely to be high during these wee days so chances of them getting cancelled due to
#          scarcity of cabs could be there

plt.hist(cabs_training.from_weekday,color='green',bins=30)
plt.xticks(range(0,7))
plt.ylabel('Frequency')
plt.xlabel('week')
plt.show()
week= cabs_training.groupby(['from_weekday', 'Car_Cancellation']).size().unstack(1)
week.head()
week.plot(kind='bar')


# In[ ]:


#DONE WITH VISUALIZATION


#!/usr/bin/env python
# coding: utf-8

# In[1]:

#BEGIN WITH DATA PREPROCESSING

import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, linear_model, metrics 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.feature_selection import SelectKBest,RFE,RFECV
from sklearn.feature_selection import chi2,f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np


# In[2]:


#noise and outliers?
#aggregation of from and to
#standardization of data
#pearson correlation and LDA for feature susetting


# In[3]:


##STARTING FEATURE SELECTION
cabs_training = pd.read_csv('Kaggle_YourCabs_training.csv')


# In[4]:


cabs_training['to_date'] = pd.to_datetime(cabs_training['to_date'])
cabs_training['from_date'] = pd.to_datetime(cabs_training['from_date'])
cabs_training['booking_created'] = pd.to_datetime(cabs_training['booking_created'])


# In[5]:


#id is set as index
cabs_training.set_index('id', inplace=True)


# In[6]:


#trying to visualize if booking and travelling date difference has any impact on Car Cancellation
booking_diff = cabs_training[['from_date', 'to_date', 'booking_created', 'Car_Cancellation']]


# In[7]:


booking_diff.loc[:, 'difference'] = booking_diff.loc[:, ('from_date')] - booking_diff.loc[:, ('booking_created')]


# In[8]:


booking_diff.head()


# In[9]:


from datetime import datetime
def convert_to_days(x):
    days = round(x.seconds / ( 24 * 60 * 60 ),5)
    return days
booking_diff.loc[:, 'diff_in_days'] = booking_diff.difference.map(convert_to_days)


# In[10]:


booking_diff.head()


# In[11]:


#AGGREGATION:As part of aggregation we are merging from_date and booking_date as days_before_booking
cabs_training.loc[:, 'days_before_booking'] = booking_diff['diff_in_days']


# In[12]:


#FEATURE CREATION :Adding addituonal columns for tracking month and week of booking
cabs_training.loc[:, 'from_month'] = cabs_training.from_date.dt.month
cabs_training.loc[:, 'from_weekday'] = cabs_training.from_date.dt.weekday

cabs_training.loc[:, 'booking_month'] = cabs_training.booking_created.dt.month
cabs_training.loc[:, 'booking_weekday'] = cabs_training.booking_created.dt.weekday


# In[13]:


cabs_training.columns


# In[15]:


#Analysis:removing from_date and booking_date  and to_date as both have been aggregated to days_before_booking
#from_city_id and to_city_id are redundantand similarly from_lat and to_lat
#Removing cost of error as my analysis will not be going to consider missclassification error.That will be my future work.
features_cols = [ 'user_id', 'vehicle_model_id', 'package_id', 'travel_type_id',
       'from_area_id', 'to_area_id','from_city_id', 'to_city_id','online_booking', 'mobile_site_booking','from_lat', 'from_long', 'to_lat', 'to_long',
        'days_before_booking', 'from_month', 'from_weekday',
       'Car_Cancellation','booking_month','booking_weekday']

#CORRELATION:visualizing the correlation between attributes
#Correlation Matrix with Heatmap
#get correlations of each features in dataset
#Analysis:the data is non multicolllinear as by heatmap we can see there is no significant correlation between the attributes
heatmap=cabs_training[features_cols]
corrmat = heatmap.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sb.heatmap(heatmap[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#Analysis: As Car_Cancellation attributes is least correlated with to_lat and to_lat is highly correlated to to_long
#          will be removing those attribute


# In[16]:


features_cols = [ 'user_id', 'vehicle_model_id', 'package_id', 'travel_type_id',
       'from_area_id', 'to_area_id','from_city_id', 'to_city_id','online_booking', 'mobile_site_booking',
        'days_before_booking', 'from_month', 'from_weekday','booking_month','booking_weekday']
X= cabs_training[features_cols]
y= cabs_training.Car_Cancellation 

#checking for missing values
X.isnull().sum()
#dealing with missing values, by creating a new catogory to define the presence of noise
X.package_id.fillna(9999, inplace=True)
X.from_area_id.fillna(9999, inplace=True)
X.to_area_id.fillna(9999, inplace=True)
X.from_city_id.fillna(9999, inplace=True)
X.to_city_id.fillna(9999, inplace=True)
X.isnull().sum()


# In[17]:


#apply SelectKBest class to extract top 10 best features
#other fetures are chi2/f-classif(ANOVA F-value)/f-regression
#f-regression is for numeric target so not using it
#Analysis of variance
bestfeatures = SelectKBest(score_func=f_classif, k=5)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(20,'Score'))


# In[18]:


#Boruta package
#Setting up 
from sklearn.ensemble import RandomForestClassifier
from boruta import boruta_py
y_nump=y.to_numpy()
X_nump=X.to_numpy()
forest = RandomForestClassifier(n_estimators="auto", max_depth=5, random_state=0, n_jobs=-1)
feat_selector = boruta_py.BorutaPy(forest, n_estimators="auto", verbose=2,perc=95,max_iter=10)
feat_selector.fit(X_nump,y_nump)
feat_selector.support_
feat_selector.ranking_


# In[20]:


#FEATURE IMPORTANCE
#travel_type_id and package_id are here very least important fetaures
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# In[21]:


#the model
model=linear_model.LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, shuffle=True)

#RFE  method i.e recursive feature elimination method without cross validation
rfe=RFE(estimator=model,step=1)
rfe=rfe.fit(X_train,y_train)

selected_rfe_features=pd.DataFrame({'Feature':list(X_train.columns),'Ranking':rfe.ranking_})
selected_rfe_features.sort_values(by='Ranking')


# In[22]:


#Optimal number of features using recursive feature elimination
print("Optimal number of features : %d" % rfe.n_features_)


# In[23]:


#4. RFECV  method i.e recursive feature elimination method with cross validation
rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')
rfecv.fit(X_train, y_train)
selected_rfe_features=pd.DataFrame({'Feature':list(X_train.columns),'Ranking':rfecv.ranking_})
selected_rfe_features.sort_values(by='Ranking')


# In[24]:


features_cols = [ 'vehicle_model_id', 'package_id', 'travel_type_id',
       'from_area_id', 'to_area_id','online_booking', 'mobile_site_booking',
        'days_before_booking', 'from_month', 'from_weekday','booking_month','booking_weekday']


# In[25]:


#END Here

#DONE WITH DATA PREPROCESSING

#BEGIN WITH DATA CLASSIFIACTION

#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, linear_model, metrics 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.feature_selection import SelectKBest,RFE,RFECV
from sklearn.feature_selection import chi2,f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,roc_auc_score,roc_curve
import numpy as np


# In[31]:


##STARTING FEATURE SELECTION
cabs_training = pd.read_csv('Kaggle_YourCabs_training.csv')


# In[32]:


# convert from_date column to date-time as it is given as object type
# Note: when we try to convert NaN to date-time, it gets converted to NaT( Not a timestamp )
#date_column is given non date time format so has been converted to date time format
# convert to_date column to date-time as it is given in float type
# convert booking_created column to date-time as it is given in float type
cabs_training['from_date'] = pd.to_datetime(cabs_training['from_date'])
cabs_training['to_date'] = pd.to_datetime(cabs_training['to_date'])
cabs_training['booking_created'] = pd.to_datetime(cabs_training['booking_created'])


# In[33]:


#id is set as index
cabs_training.set_index('id', inplace=True)


# In[34]:


#trying to visualize if booking and travelling date difference has any impact on Car Cancellation
booking_diff = cabs_training[['from_date', 'to_date', 'booking_created', 'Car_Cancellation']]
booking_diff.loc[:, 'difference'] = booking_diff.loc[:, ('from_date')] - booking_diff.loc[:, ('booking_created')]
from datetime import datetime
def convert_to_days(x):
    days = round(x.seconds / ( 24 * 60 * 60 ),5)
    return days
booking_diff.loc[:, 'diff_in_days'] = booking_diff.difference.map(convert_to_days)


# In[35]:


#As part of aggregation we are merging from_date and booking_date as days_before_booking
cabs_training.loc[:, 'days_before_booking'] = booking_diff['diff_in_days']
cabs_training.loc[:, 'from_month'] = cabs_training.from_date.dt.month
cabs_training.loc[:, 'from_weekday'] = cabs_training.from_date.dt.weekday

cabs_training.loc[:, 'booking_month'] = cabs_training.booking_created.dt.month
cabs_training.loc[:, 'booking_weekday'] = cabs_training.booking_created.dt.weekday


# In[36]:


#wrapper methods is costly as subset of all attributes are calcluated accuracy against a model.
#embedded menthods are less costy and avoid overfiiting
#corre;ation between attributes is found out to check the relevance
#need to remove redunat and irrevalnt features so willl use feature subset selction
#MOre numbers of features are likely to have high accuracy but low interpretion and chances of getting overfit
#regression: used when a dependemt variable need to be predicted from independent variable.The dependent variale is continuous in type.
#the realtion between the dependented and independet variable is linear in nature, i.e as we change ind variable the dependent var changes.
#the value of predicted value can be discrete(0/1) or a value(sales of company)
#regression coefficnt tells how much a dpendepnt variable is imporatnt in predicting the dependent value
#using logisitic regression since it gives the best model that can show relationship between dependent and independent 
#linear regression is important if relationship is linear in nature
#logistic regression had binary dependent variable value .Independent var can be descrite or continus.Probability is used for setup the realtionships.
#sigmoid fucntion will convert a number to be in range from 0 to 1
#model.fit is basically training the model
#model.score shows accuracy
#Filter menthods: correlation: predictors should have hogh correletoon with target and low correlation between predictors


# In[37]:


#BEGIN TEMPORARY CLASSIFICATION 


# In[38]:


#We test which classifier to use.Trainig set is divide into test and training.
#Supervised learning is the machine learning task where the training data has been lablled , And since we need to maap
#test data to predfined classes, it is a classification task


# In[39]:


features_cols = ['vehicle_model_id', 'travel_type_id', 'from_area_id',
                 'to_area_id','from_month', 'from_weekday',
                 'days_before_booking', 'online_booking', 'mobile_site_booking',
                 'booking_month', 'booking_weekday','Car_Cancellation','package_id']
df_class=cabs_training[features_cols]
df_class.from_area_id.fillna(-9999, inplace=True)
df_class.to_area_id.fillna(-9999, inplace=True)
df_class.package_id.fillna(-9999, inplace=True)


# In[40]:


Xt = df_class[features_cols]
yt = df_class.Car_Cancellation
Xtr, Xv, ytr, yv = train_test_split(Xt, yt, test_size=0.3)


# In[41]:


classifiers=[]
model1 = svm.SVC()
classifiers.append(model1)
model3 = RandomForestClassifier()
classifiers.append(model3)


# In[43]:


#out of 885(10)+64(11) only 64 is correctly predicted leading to recall 0.07 which is very low
#out of the correct prediction only 75 % of the prediction is correct leaidng to precision
sss = StratifiedShuffleSplit(random_state=0, n_splits=2, test_size=0.3)
for clf in classifiers:
    print(clf.__class__.__name__)
    clf.fit(Xtr, ytr)
    y_pred= clf.predict(Xv)
    acc = accuracy_score(yv, y_pred)
    print(np.unique(y_pred, return_counts=False))
    print("Accuracy is",acc)
    cm = confusion_matrix(yv, y_pred)
    print("Confusion Matrix is",cm)
    print("Report",classification_report(yv,y_pred))
    cv_scores = cross_val_score(clf, Xtr, ytr, scoring='roc_auc', cv=sss)
    print("CV Scores",cv_scores.mean(),clf)
    


# In[44]:


df_class['Car_Cancellation'].value_counts()


# In[45]:


df_majority=df_class[df_class.Car_Cancellation==0]
df_minority=df_class[df_class.Car_Cancellation==1]
from sklearn.utils import resample
df_minority_unsampled=resample(df_minority,replace=True,n_samples=40299,random_state=123)
df_upsampled=pd.concat([df_majority,df_minority_unsampled])
df_upsampled['Car_Cancellation'].value_counts()


# In[46]:


Xt = df_upsampled[features_cols]
yt = df_upsampled.Car_Cancellation
Xtr, Xv, ytr, yv = train_test_split(Xt, yt, test_size=0.3)
sss = StratifiedShuffleSplit(random_state=0, n_splits=2, test_size=0.3)
for clf in classifiers:
    clf.fit(Xtr, ytr)
    y_pred= clf.predict(Xv)
    acc = accuracy_score(yv, y_pred)
    print(np.unique(y_pred, return_counts=False))
    print("Accuracy of %s is %s"%(clf, acc))
    cm = confusion_matrix(yv, y_pred)
    print("Confusion Matrix of %s is %s"%(clf, cm))
    print("Report",classification_report(yv,y_pred))
    cv_scores = cross_val_score(clf, Xtr, ytr, scoring='roc_auc', cv=sss)
    print(cv_scores.mean(),clf)


# In[47]:


df_majority_sec=df_class[df_class.Car_Cancellation==0]
df_minority_sec=df_class[df_class.Car_Cancellation==1]
df_majority_downsampled=resample(df_majority_sec,replace=False,n_samples=3132,random_state=123)
df_downsampled=pd.concat([df_majority_downsampled,df_minority_sec])
df_downsampled['Car_Cancellation'].value_counts()


# In[48]:


Xt = df_upsampled[features_cols]
yt = df_upsampled.Car_Cancellation
Xtr, Xv, ytr, yv = train_test_split(Xt, yt, test_size=0.3)
sss = StratifiedShuffleSplit(random_state=0, n_splits=2, test_size=0.3)
for clf in classifiers:
    clf.fit(Xtr, ytr)
    y_pred= clf.predict(Xv)
    acc = accuracy_score(yv, y_pred)
    print(np.unique(y_pred, return_counts=False))
    print("Accuracy of %s is %s"%(clf, acc))
    cm = confusion_matrix(yv, y_pred)
    print("Confusion Matrix of %s is %s"%(clf, cm))
    print("Report",classification_report(yv,y_pred))
    cv_scores = cross_val_score(clf, Xtr, ytr, scoring='roc_auc', cv=sss)
    print(cv_scores.mean(),clf)





#END WITH DATA CLASSIFICATION



