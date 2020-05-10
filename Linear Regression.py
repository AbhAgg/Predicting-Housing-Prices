import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
MyData=pd.read_csv("D:\Python classes\houseRent\houseRent\housing_train.csv")

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

#Dropping the Latitude column as there are very less values
x.dropna(subset = ["lat"], inplace=True)

#Dropping the Description column as there are very less values
x.dropna(subset = ["description"], inplace=True)

#Filling the Laundry options column with No Info to make it another category
x=x.fillna({'laundry_options':'noinfo'})

#Filling the Parking Options column with No Info to make it another category
x=x.fillna({'parking_options':'noinfo'})

#Checking again for any null values both figuratively and via count
sns.heatmap(x.isnull(),yticklabels=False,cbar=False,cmap='viridis')
count=x.isna().sum()

###Now the data has no NULL Values###



######Outlier Identification#######
#####Checking for outliers#########

x.describe()


sns.boxplot(x[])
MyData['price'].plot()
plt.hist(MyData['price'].values)

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(MyData['sqfeet'],MyData['price'])
plt.show()

########Using Z-Scores to find outliers######

##### We can also use SCIPY for stats
'''
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(MyData['price']))
print(z)
threshold=3
print(np.where(z>3))

mydata_clean = MyData
mydata_clean = MyData[(z<3).all(axis=1)]

'''
######

def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outlier_datapoints = detect_outlier(MyData['price'])
print(outlier_datapoints)

######CLEANING DATA (Removing outliers)






###########Using IQR to find outliers########
a=sorted(MyData['price'])
q1, q3= np.percentile(a,[25,75])



iqr = q3 - q1
lower_bound = q1 -(1.5 * iqr) 
upper_bound = q3 +(1.5 * iqr)

#OR
#Q1=MyData['price'].quantile(0.25)
#Q3=MyData['price'].quantile(0.75)
#IQR=Q3-Q1
#print(IQR)
#print(MyData['price']>(Q1-1.5*IQR)) and (MyData['price']<(Q3+1.5*IQR))

######CLEANING DATA (Removing outliers)