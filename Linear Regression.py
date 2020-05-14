import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy import stats
MyData=pd.read_csv("D:\Python classes\houseRent\houseRent\housing_train.csv")
MyData.dtypes
#### You can check the type of the Dataframe
####You can use fillna with 0 and then check the count
######Visualising the data#####

#Checking for Null Values graphically
sns.heatmap(MyData.isnull(),yticklabels=False,cbar=False,cmap='viridis')


#Checking the distribution of laundry in this case
sns.set_style('whitegrid')
sns.countplot(x='laundry_options',data=MyData,palette='RdBu_r')

#Counting the NULL/NAN values in each column to rectify them
count=MyData.isna().sum()


#Keeping the original dataset untouched and transferring it to x
x=MyData

#Dropping the null values in Latitude column as there are very less values
x.dropna(subset = ["lat"], inplace=True)

#Dropping the null valuesDescription column as there are very less values
x.dropna(subset = ["description"], inplace=True)

#Filling the Laundry options column with No Info to make it another category
x=x.fillna({'laundry_options':'noinfo'})

#Filling the Parking Options column with No Info to make it another category
x=x.fillna({'parking_options':'noinfo'})

#Checking again for any null values both figuratively and via count
sns.heatmap(x.isnull(),yticklabels=False,cbar=False,cmap='viridis')
count=x.isna().sum()

###Now the data has no NULL Values###

########## Feature Enginering ############

#Checking for unique/distinct values in each independent variable.
#This will help to decide which ones need dummy variable treatment and not
#

# Finding the data type of each column
x.dtypes
for col in x.columns:
    print(col.ljust(30),': ',len(x[col].unique()),'labels'.ljust(10),' : ',x[col].dtype)



####Making of a duplicate dataset to prevent overlapping 

##################################################### LAUNDRY ###################
v=x
####Making the dummy variable for LAUNDRY_OPTIONS
dummies = pd.get_dummies(v.laundry_options)
#Merge the data
merged = pd.concat([v,dummies],axis='columns')
merged.shape
merged = merged.drop('laundry_options',axis=1)
merged = merged.drop('noinfo',axis=1)
list(merged.columns)

for col in merged.columns:
    print(col.ljust(30),': ',len(merged[col].unique()),'labels'.ljust(10),' : ',merged[col].dtype)


##### Find the count of unique/individual labes in a particular feature ####


#################################################### PARKING ################# 

####Making the dummy variable for PARKING_OPTIONS
dummies = pd.get_dummies(merged.parking_options)
merged = pd.concat([merged,dummies],axis='columns')
merged = merged.drop('parking_options',axis=1)
merged = merged.drop('noinfo',axis=1)
list(merged.columns)

for col in merged.columns:
    print(col.ljust(30),': ',len(merged[col].unique()),'labels'.ljust(10),' : ',merged[col].dtype)


######Outlier Identification#######
#####Checking for outliers#########

#Dropping the description Column for now


#Visualising the particular feature for outliers


#############################################   BEDS     ###########

merged.beds.value_counts().sort_values(ascending=False).head(20)



merged1 = merged

sns.boxplot(merged1['beds'])
merged1['beds'].plot()
plt.hist(merged1['beds'].values)

fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(merged1['beds'],merged1['price'])
plt.show()


###Counting the value of each type of beds
merged1.beds.value_counts()

##Manually removing the outliers###
merged1=merged1[merged1['beds']<300]

###Counting again the value of each type of beds
merged1.beds.value_counts()



'''

NOT NEEDED HERE


########Using Z-Scores to find outliers######

##### We can also use SCIPY for statsf

z_scores = scipy.stats.zscore(merged['beds'])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
merged = merged[filtered_entries]
'''



#############################################   BATHS   ############

sns.boxplot(merged1['baths'])
merged1['baths'].plot()
plt.hist(merged1['baths'].values)

fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(merged1['baths'],merged1['price'])
plt.show()


###Counting the value of each type of baths
merged1.baths.value_counts()
merged1=merged1[merged1['baths']<5]
merged1.baths.value_counts()

##### No need to eliminate any outliers (But there is a scope of improvement)

'''
########Using Z-Scores to find outliers######

##### We can also use SCIPY for stats

z_scores = scipy.stats.zscore(merged['baths'])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
merged = merged[filtered_entries]

'''

'''
outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outlier_datapoints = detect_outlier(merged['beds'])
print(outlier_datapoints)
'''



'''
###### MODIFIED Z Score Method #########

def outliers_modified_z_score(ys):
    threshold = 3.5
    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    
    for y in ys:
        modified_z_scores = 0.6745 * (y - median_y) / median_absolute_deviation_y
        if(np.abs(modified_z_scores) > threshold):
            outliers.append(y)
    return outliers


outlier_datapoints = detect_outlier(qwert['sqfeet'])
print(outlier_datapoints)
'''



'''
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))
'''


#############################################   IMAGE URL   ############


# Creating an addition boolean column for presence of images
merged1['bool_imageurl'] = 1

#dropping the image_url column 
merged1 = merged1.drop('image_url',axis=1)


#############################################   Description   ############


# Creating an addition boolean column for presence of description
merged1['bool_description'] = 1

#dropping the description column 
merged1 = merged1.drop('description',axis=1)


##############################################   PRICE ########################

sns.boxplot(merged1['price'])
merged1['price'].plot()
plt.hist(merged1['price'].values)

fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(merged1['price'],merged1['price'])
plt.show()

merged1=merged1[merged1['price']<7000]

merged1=merged1[merged1['price']>100]
merged1.price.value_counts()

##############################################   PRICE PER SQFEET ########################
price_per_sqfeet = merged1['price']/merged1['sqfeet']
price_per_sqfeet=price_per_sqfeet.to_frame(name="val")
price_per_sqfeet = price_per_sqfeet.rename(columns = {"val":"price_per_sqfeet"})
merged1 = pd.concat([merged1,price_per_sqfeet],axis=1)

############################################## ID ########################

## Dropping the ID column

merged1 = merged1.drop('id',axis=1)



############################################## URL ########################

## Dropping the URL column

merged1 = merged1.drop('url',axis=1)



############################################## Region URL ########################

## Dropping the Region URL column

merged1 = merged1.drop('region_url',axis=1)



#############################################   SQFEET AREA  ############

sns.boxplot(merged1['sqfeet'])
merged1['sqfeet'].plot()
plt.hist(merged1['sqfeet'].values)

merged1.sqfeet.value_counts()
temp = sorted(merged1['sqfeet'], reverse=True)

merged1=merged1[merged1['sqfeet']<3300]

merged1=merged1[merged1['sqfeet']>100]


###########Using IQR to find outliers########
'''
#### Percentile Method   ####
sns.boxplot(merged['sqfeet'])
qwe=[]
q1, q3= np.percentile(merged['sqfeet'],[25,75])
iqr = q3 - q1
lower_bound = q1 -(1.5 * iqr) 
upper_bound = q3 +(1.5 * iqr)

'''

####### Quantile Method  ####
merged1=merged
Q1=merged['sqfeet'].quantile(0.25)
Q3=merged['sqfeet'].quantile(0.75)
IQR=Q3-Q1
LB = Q1-(1.5*IQR)
UB = Q3+(1.5*IQR)


merged1= merged[merged['sqfeet'].between(LB, UB)]
sns.boxplot(merged1['sqfeet'])

##########################################   TYPE and TYPE MEAN ENCODING  ##################


mean_encode = merged1.groupby('type')['price'].mean()
merged1.loc[:,'type_mean_enc'] = merged1['type'].map(mean_encode)
merged1 = merged1.drop('type',axis=1)


####################   Saving to a CSV #################
# merged2.to_csv('D:\\Python classes\\houseRent\\houseRent\\trew.csv',columns=['price','type'])

######################################## LONGITUDE ##################

merged1=merged1[merged1['long']>-130]
plt.hist(merged1['long'].values)

#### Colored Plot between longitude and latitude ####
merged1.plot(kind="scatter", x="long", y="lat", alpha=0.4, figsize=(10,7),
    c="price", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)

merged1 = merged1.drop('long',axis=1)



# Helped us visualise that we can distribute the states and regions price/sqfeet  between 
######################################## LATITUDE ##################

merged1=merged1[merged1['lat']<50]
merged1=merged1[merged1['lat']>20]

plt.hist(merged1['lat'].values)

merged1 = merged1.drop('lat',axis=1)

##########################################   STATE and STATE MEAN ENCODING  ##################

mean_encode = merged1.groupby('state')['price_per_sqfeet'].mean()
merged1.loc[:,'state_mean_encoding'] = merged1['state'].map(mean_encode)
merged1 = merged1.drop('state',axis=1)


##########################################   REGION and REGION MEAN ENCODING  ##################

mean_encode = merged1.groupby('region')['price_per_sqfeet'].mean()
merged1.loc[:,'region_mean_encoding'] = merged1['region'].map(mean_encode)
merged1 = merged1.drop('region',axis=1)





