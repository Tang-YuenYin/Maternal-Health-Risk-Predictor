#!/usr/bin/env python
# coding: utf-8

# # WIA1006 Machine Learning
# # Healthcare Assignment (20%)
# # Title : Prediction of Martenal Health Risk 
# # Group Name: Help Our Mother

# # Members:
# ### 1. Florence Wong Leh Yi 22004751/1
# ### 2. Ong Yee Shuen 22004823/1
# ### 3. Toh Yan Xin 22052198/1
# ### 4. Tay Min En 22004829/1
# ### 5. Poon Chuan Rich 22059181/1

# # Introduction :
# Maternal health plays a crucial role in the well-being of both mothers and their infants. Ensuring the health and safety of expectant mothers is a significant concern in healthcare. In this assignment, we aim to develop a predictive model that can assess the risk of maternal health complications, enabling early detection and intervention. 
# Predicting maternal health risk can help healthcare providers identify high-risk cases and allocate appropriate resources and interventions to mitigate potential complications. By leveraging machine learning techniques and analyzing relevant features, such as demographic information, medical history, and prenatal care data, we can build a model that predicts the likelihood of health risks for expectant mothers.
# 
# The objective of this assignment is to utilize a dataset containing information about maternal health, pregnancy, and related factors to develop a predictive model. By exploring the dataset, performing data wrangling and preprocessing, and applying various regression or classification algorithms, we can create a model that accurately predicts maternal health risk. The assignment will also involve evaluating the model's performance and discussing its potential implications in clinical practice.
# 
# Through this assignment, we aim to contribute to the advancement of maternal healthcare by harnessing the power of data analysis and predictive modeling. By identifying and addressing maternal health risks proactively, we can improve outcomes for mothers and infants, reducing complications and ensuring a healthier pregnancy journey.
#     

# # Feature Introduction 

# # Age:
# Age refers to the chronological age of an individual, measured in years.
# 
# Normal Range: 
# Age is a continuous feature and does not have a specific normal range. However, it is important to consider age-related health risks and consult appropriate medical guidelines for different age groups.
# 
# ## Systolic Blood Pressure (SBP):
# Systolic blood pressure is the pressure exerted on the arteries when the heart contracts and pumps blood.
# 
# Normal Range: The normal range for systolic blood pressure is typically below 120 mmHg. Values between 120-129 mmHg are categorized as elevated, while values of 130 mmHg or higher are considered high blood pressure (hypertension).
# 
# ## Diastolic Blood Pressure (DBP):
# Diastolic blood pressure is the pressure exerted on the arteries when the heart is at rest between beats.
# 
# Normal Range: The normal range for diastolic blood pressure is typically below 80 mmHg. Values between 80-89 mmHg are categorized as elevated, while values of 90 mmHg or higher are considered high blood pressure (hypertension).
# 
# ## Blood Sugar (BS):
# Blood sugar, also known as blood glucose, refers to the concentration of glucose in the bloodstream.
# 
# Normal Range: The normal fasting blood sugar range is generally between 70-99 mg/dL (milligrams per deciliter). However, blood sugar levels can vary depending on factors such as time of day, meals, and individual circumstances. It is important to consult with a healthcare professional for specific guidelines based on individual health conditions.
# 
# ## Body Temperature:
# Body temperature refers to the internal temperature of the body.
# 
# Normal Range: The normal range for body temperature is typically considered to be around 98.6°F (37°C). However, normal body temperature can vary slightly between individuals, and it can be influenced by factors such as age, activity level, and time of day.
# 
# ## Heart Rate:
# Heart rate is the number oftimes the heart beats per minute.
# 
# Normal Range: The normal resting heart rate for adults is generally between 60-100 beats per minute. However, it can vary depending on factors such as age, fitness level, and overall health.
# 
# ## Risk Level:
# Risk level refers to the categorization or assessment of the level of risk associated with a particular condition or outcome.
# 
# Normal Range: Risk level is typically evaluated based on specific criteria and may vary depending on the context of the analysis. It is determined based on factors such as statistical thresholds, clinical guidelines, or predetermined thresholds established for a specific risk assessment model.
# 
# 
# These normal ranges serve as general guidelines, and it is important to consider individual factors, medical history, and consult with healthcare professionals for accurate interpretation and personalized assessment.

# In[1]:


get_ipython().system('pip install catboost')


# In[2]:


pip install imblearn


# In[3]:


pip install dataprep


# In[4]:


get_ipython().system('pip install --upgrade scikit-learn')


# # Import Library

# In[5]:


#Data Wrangling
import pandas as pd 
import numpy as np

#Ignore warnigs to improve code readability
import warnings
warnings.filterwarnings("ignore")

#Data visualization
import matplotlib.pyplot as plt # plotting library
from dataprep.eda import plot
import seaborn as sns # additional functionality and enhancements

#Data preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

#Regression models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor
from xgboost import XGBClassifier
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge

#Imbalance data processing
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# # Introduction to dataset

# # Import Dataset in csv file

# In[6]:


#Read the CSV file into a DataFrame
data = pd.read_csv('Maternal Health Risk Data Set.csv')
print("\033[1mDataset of Martenal Health Data Set : \033[0m")
data


# # Display the first 10 roles of data as heading

# In[7]:


# Check the head of the dataset 
print("\033[1mFirst 10 rows of the data : \033[0m")
data.head(10)


# # Display the last 10 rows as tail

# In[8]:


# Check the tail of the dataset
print("\033[1mLast 10 rows of the data : \033[0m")
data.tail(10)


# # Obtain Basic Information
# ### Columns, Datatypes and memory usages

# In[9]:


print("\033[1mInformation of the data :\033[0m ")
data.info()


# # Description of dataset

# In[10]:


# Describe the data 
# Count, mean, min, max, 25%, 50%, 75%, std
print("\033[1mDescription of Data : \033[0m")
data.describe().T


# In[11]:


# Check dimensionally of the DataFrame
print("\033[1mShape of Data :\033[0m")
data.shape


# # Data Visualization and Exploration

# In[12]:


# Plot pairwise relantionships 
# Pair of Target with all other features in dataset
sns.pairplot(data, hue = 'RiskLevel')
plt.show()


# In[13]:


# Plot heatmap
# making visualization of patterns, to identify correlations between variable
plt.figure(figsize=(15,12))
sns.heatmap(data.corr(), annot=True)
plt.show()


# # Count of RiskLevel based on Age

# In[14]:


# Counts of RiskLevel for each Age
plt.figure(figsize=(10, 6))  # Increase the figure size

sns.countplot(x='Age', hue='RiskLevel', data=data)
plt.title('Counts of RiskLevel for each Age')  # Add a title to the plot
plt.xlabel('Age')  # Set the x-axis label
plt.ylabel('Count')  # Set the y-axis label

# Adjust the appearance of the plot
sns.set(style="whitegrid")  # Set the plot style
sns.despine()  # Remove the top and right spines

plt.tight_layout()  # Adjust the spacing of the plot
plt.show()


# In[15]:


# Create a table of counts
count_table = data.groupby(['Age', 'RiskLevel']).size().unstack().fillna(0)

# Customize the table display
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Add some styling
count_table_style = count_table.style.background_gradient(cmap='Blues')

# Display the table with styling
print("\033[1mTable of Maternal Health Data Set Counts:\033[0m")
count_table_style


# # Check the correlation

# In[16]:


correlation = data[['RiskLevel', 'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'HeartRate']].corr()
correlation


# Age has a positive correlation with SystolicBP (0.416), DiastolicBP (0.398), and BS (0.473). This suggests that as age increases, there is a tendency for these variables to increase as well, although the correlations are not very strong.
# 
# SystolicBP and DiastolicBP have a strong positive correlation of 0.787. This indicates that as SystolicBP increases, DiastolicBP also tends to increase.
# 
# DiastolicBP and BS have a positive correlation of 0.424. This suggests that as DiastolicBP increases, there is a tendency for BS to increase as well, although the correlation is relatively weak.
# 
# BS has a positive correlation with Age (0.473) and SystolicBP (0.425). This indicates that as BS increases, there is a tendency for Age and SystolicBP to increase, although the correlations are not very strong.
# 
# HeartRate has a weak positive correlation with Age (0.080) and BS (0.143), suggesting that as Age and BS increase, HeartRate also tends to increase, although the correlations are relatively weak.
# 
# HeartRate has weak negative correlations with SystolicBP (-0.023) and DiastolicBP (-0.046). This indicates that as SystolicBP and DiastolicBP increase, HeartRate tends to slightly decrease, although the correlations are weak.

# In[17]:


data["RiskLevel"].value_counts()
# Count the unique value of the RiskLevel
# Unique Value : low risk, mid risk, high risk


# In[18]:


# Show Pie Chart and Bar Chart of RiskLevel 
fig = plt.figure(figsize=(5, 5))

# Define custom colors for the pie chart
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

# Generate the pie chart with custom colors and shadow
plt.pie(data['RiskLevel'].value_counts(), labels=list(data['RiskLevel'].unique()), autopct='%1.1f%%',
        colors=colors, shadow=True)

# Add a title with a larger font size and bold text
plt.title('RiskLevel Distribution', fontsize=16, fontweight='bold')

# Add a legend with smaller font size and adjust its position
plt.legend(loc='upper right', fontsize=10)

# Adjust the appearance of the plot
plt.axis('equal')
plt.tight_layout()  # Improve spacing between elements

plt.show()

plt.figure(figsize=(6, 8))
sns.countplot(x='RiskLevel', data=data, palette='viridis')
plt.xlabel('Risk Level')
plt.ylabel('Count')
plt.title('Distribution of Risk Levels')
plt.xticks(rotation=0)
plt.show()


# # Replace the target variable 
# # String to numeric
# # 0 (low risk)
# # 1 (mid risk)
# # 2 (high risk)

# In[19]:


# replace the target variable from string to numeric
data['RiskLevel'] = data['RiskLevel'].replace({'low risk':0, 'mid risk':1, 'high risk':2})
data 
# Risk Level Values are changed to 0(low risk), 1(mid risk) ,2(high risk)

# Data type has been changed from object to int64 (for RiskLevel)
data.info()


# # Change the body temperature from Fahrenhit(°F) to Celcius (°C)
# ## Celsius is often considered more useful than Fahrenheit
# ## More easier for the users to input their temperatures

# In[20]:


# Convert the BodyTemp (Body Temperature) from F (Fahrenheit) to C (Celcius)
data['BodyTemp_C'] = (data['BodyTemp'] - 32) * 5/9
data['BodyTemp'] = data['BodyTemp_C']
data = data.drop(columns=['BodyTemp_C']) # drop the extra columns to prevent duplicate data
data


# After change BodyTemp from F to C, we can observed that the temperature are in infinity decimal 

# In[21]:


# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the histogram
data.hist(ax=ax)

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()


# In[22]:


# Check correlation between all feature after conversion of target variable RiskLevel to numeric
plt.figure(figsize=(15,12))
sns.heatmap(data.corr(), annot=True)
plt.show()


# # Overview for eachdata

# In[23]:


# using dataprep class to show Statistics, Histogram, KDE plot, Normal Q-Q plot, Box Plot, Value Tabale 
plot(data,'Age') 
# We have data of age ranges from 10 years old to 70 years old


# In[24]:


plot(data, 'BS')
# We have the BS ranges from 6 to 19
# Value after 8 consider high risk, they are valid and not consider outlier


# In[25]:


plot(data, 'BodyTemp')
# Body Temp range of 36.88 to 39


# In[26]:


plot(data,'DiastolicBP')
# The range of DiastolicBP is from 49 to 100


# In[27]:


plot(data,'HeartRate')


# In[28]:


# Customize the line plot with smooth lines
sns.regplot(x='DiastolicBP', y='SystolicBP', marker='o', scatter_kws={'s': 50}, ci=None, data=data, lowess=True, line_kws={'linewidth': 2, 'color': 'orange'})

# Set plot labels
plt.title('Diastolic BP vs Systolic BP', fontsize=16)
plt.xlabel('Diastolic Blood Pressure', fontsize=12)
plt.ylabel('Systolic Blood Pressure', fontsize=12)

# Customize the grid
ax.grid(color='#dddddd', linestyle='-', linewidth=0.5)

# Customize the background color
ax.set_facecolor('#f9f9f9')

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()


# ###  indicates when DBP and SBP is high then Risk is high 

# In[29]:


# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Customize the line plot
sns.lineplot(x='DiastolicBP', y='SystolicBP', hue='RiskLevel', data=data, palette=['green', 'orange', 'red'])
plt.title('Diastolic BP vs Systolic BP by Risk Level')
plt.xlabel('Diastolic Blood Pressure')
plt.ylabel('Systolic Blood Pressure')
plt.legend(title='Risk Level')

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()


# In[30]:


grid = sns.FacetGrid(data, col="RiskLevel", hue="RiskLevel", col_wrap=3, palette=['green', 'orange', 'red'])
grid.map(sns.lineplot, "DiastolicBP", "SystolicBP")
grid.add_legend()
plt.show()


# In[31]:


# below scatter plot indicates total count of high risklevel in BS is high compared to low RiskLevel
# So BS is impactful factor for high risk  
sns.scatterplot(x='RiskLevel', y='BS', hue='RiskLevel', data=data, palette=['green', 'orange', 'red'])
plt.show()


# In[32]:


# As BS increases, Risk also increases based on Age
# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Customize the line plot
sns.lineplot(x='Age', y='BS', hue='RiskLevel', data=data, palette=['#63c381', '#ffa94d', '#e74c3c'], linewidth=2)

# Set plot labels and title
plt.title('Age vs Blood Sugar (BS) by Risk Level', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Blood Sugar (BS)', fontsize=12)

# Customize the legend
plt.legend(title='Risk Level', title_fontsize=12)

# Customize the grid
ax.grid(color='#dddddd', linestyle='-', linewidth=0.5)

# Customize the background color
ax.set_facecolor('#f9f9f9')

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()


# ### BS increases, Risk also increases based on Age

# # Preprocessing or Feature Engineering

# In[33]:


# Count Missing Value
print("\033[1mMissing values by Column : \033[0m")
print("-"*30)
print(data.isna().sum())
print("-"*30)
print("Total Missing Values: ",data.isna().sum().sum())

# No missing values 


# ### No Missing value

# # Split the data into feature and target set
# # Target is RiskLevel, which already converted into from string to numeric

# In[34]:


# Split data into feature and target
X = data.drop('RiskLevel',axis = 1)
y = data['RiskLevel']
print("X Shape : ", X.shape)
print("y Shape : ", y.shape)


# # MinMaxScaler and StandardScaler

# ### Both techniques used to transform numerical data

# ### MinMaxScaler:
# 
# -Scales the features to a specific range, typically between 0 and 1.
# 
# -Preserves the original distribution shape of the data.
# 
# -Useful when the features have a bounded range and when preserving the relationships between the features is important.
# 
# -Can be sensitive to outliers if the range is heavily influenced by extreme values 

# ### StandardScaler:
# 
# -Transforms the features to have zero mean and unit variance.
# 
# -Centers the data around 0, making the mean of each feature 0 and standard deviation of 1.
# 
# -Does not bound the range of the features.
# 
# -Works well when the data does not have a specific range requirement and when the algorithm used for modeling assumes normally distributed features.
# 
# -Handles outliers better as it uses the mean and standard deviation. 

# In[35]:


normal = MinMaxScaler()
standard = StandardScaler()


# ### MinMaxScaler 

# In[36]:


normalised_features = normal.fit_transform(X)
normalised_data = pd.DataFrame(normalised_features, index = X.index, columns = X.columns)


# ### StandardScaler 

# In[37]:


standardised_features = standard.fit_transform(X)
standardised_data = pd.DataFrame(standardised_features, index = X.index, columns = X.columns)


# In[38]:


# Create subplots
fig, ax = plt.subplots(1, 3, figsize=(30, 15))

# Set style
sns.set(style="ticks")

# Original
sns.boxplot(x='variable', y='value', data=pd.melt(data[X.columns]), ax=ax[0], palette='pastel')
ax[0].set_title('Original')
ax[0].set_xlabel('Variable')
ax[0].set_ylabel('Value')
ax[0].tick_params(axis='x', labelrotation=45)

# MinMaxScaler
sns.boxplot(x='variable', y='value', data=pd.melt(normalised_data[X.columns]), ax=ax[1], palette='pastel')
ax[1].set_title('MinMaxScaler')
ax[1].set_xlabel('Variable')
ax[1].set_ylabel('Value')
ax[1].tick_params(axis='x', labelrotation=45)

# StandardScaler
sns.boxplot(x='variable', y='value', data=pd.melt(standardised_data[X.columns]), ax=ax[2], palette='pastel')
ax[2].set_title('StandardScaler')
ax[2].set_xlabel('Variable')
ax[2].set_ylabel('Value')
ax[2].tick_params(axis='x', labelrotation=45)

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.3)

# Customize the background color
fig.set_facecolor('#f9f9f9')

# Remove top and right spines
for axis in ax:
    sns.despine(ax=axis)

# Remove unnecessary ticks
sns.despine(bottom=True, left=True)

# Set plot title
plt.suptitle('Boxplots of Variables', fontsize=16)

# Show the plot
plt.show()


# # Build Model
# # Regression Models

# In[39]:


scalers = [normal, standard]


# # Split the data into training and testing set

# In[40]:


# split the data into testing set and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# # Define and declare all regression mode
# 

# In[41]:


knn = KNeighborsRegressor()
svr = SVR()
tree = DecisionTreeRegressor(max_depth = 10, random_state = 42)
xgb = XGBRegressor()
catb = CatBoostRegressor()
linear = LinearRegression()
sdg = SGDRegressor()
rfr = RandomForestRegressor(max_depth=2, random_state=0)
gb = GradientBoostingRegressor(random_state=0)
br = BayesianRidge()


# ##  K-Nearest Neighbors Regression (KNN Regression)

# In[42]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score



# Instantiate scaler objects
minmax_scaler = MinMaxScaler()
std_scaler = StandardScaler()

knn = KNeighborsRegressor()
knn_rmse = []
knn_mae = []
knn_mse = []
knn_r2 = []

# Without feature scaling
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
knn_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
knn_mae.append(mean_absolute_error(y_test, pred))
knn_mse.append(mean_squared_error(y_test, pred))
knn_r2.append(r2_score(y_test, pred))

# Apply different scaling techniques and make predictions using KNN
scalers = [minmax_scaler, std_scaler]
scaler_names = ['MinMaxScaler', 'StandardScaler']

for scaler, scaler_name in zip(scalers, scaler_names):
    pipe = make_pipeline(scaler, knn)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    knn_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
    knn_mae.append(mean_absolute_error(y_test, pred))
    knn_mse.append(mean_squared_error(y_test, pred))
    knn_r2.append(r2_score(y_test, pred))

# Create a DataFrame to display the results
knn_df = pd.DataFrame({
    'Root Mean Squared Error': knn_rmse,
    'Mean Absolute Error': knn_mae,
    'Mean Squared Error': knn_mse,
    'R-squared': knn_r2
}, index=['Original'] + scaler_names)
knn_df


# ## Support Vector Regression (SVR)

# In[43]:


svr = SVR()
svr_rmse = []
svr_mae = []
svr_mse = []
svr_r2 = []

# Without feature scaling
svr.fit(X_train, y_train)
pred = svr.predict(X_test)
svr_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
svr_mae.append(mean_absolute_error(y_test, pred))
svr_mse.append(mean_squared_error(y_test, pred))
svr_r2.append(r2_score(y_test, pred))

# Apply different scaling techniques and make predictions using SVR
for scaler, scaler_name in zip(scalers, scaler_names):
    pipe = make_pipeline(scaler, svr)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    svr_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
    svr_mae.append(mean_absolute_error(y_test, pred))
    svr_mse.append(mean_squared_error(y_test, pred))
    svr_r2.append(r2_score(y_test, pred))

# Create a DataFrame to display the results
svr_df = pd.DataFrame({
    'Root Mean Squared Error': svr_rmse,
    'Mean Absolute Error': svr_mae,
    'Mean Squared Error': svr_mse,
    'R-squared': svr_r2
}, index=['Original'] + scaler_names)
svr_df


# ## XG Boost Regression
# 

# In[44]:


xgb_rmse = []
xgb_mae = []
xgb_mse = []
xgb_r2 = []

# Without feature scaling
xgb.fit(X_train, y_train)
pred = xgb.predict(X_test)
xgb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
xgb_mae.append(mean_absolute_error(y_test, pred))
xgb_mse.append(mean_squared_error(y_test, pred))
xgb_r2.append(r2_score(y_test, pred))

# Apply different scaling techniques and make predictions using XGBoost
for scaler, scaler_name in zip(scalers, scaler_names):
    pipe = make_pipeline(scaler, xgb)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    xgb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
    xgb_mae.append(mean_absolute_error(y_test, pred))
    xgb_mse.append(mean_squared_error(y_test, pred))
    xgb_r2.append(r2_score(y_test, pred))

# Create a DataFrame to display the results
xgb_df = pd.DataFrame({
    'Root Mean Squared Error': xgb_rmse,
    'Mean Absolute Error': xgb_mae,
    'Mean Squared Error': xgb_mse,
    'R-squared': xgb_r2
}, index=['Original'] + scaler_names)
xgb_df


# ## CatBoost Regression

# In[45]:


from catboost import CatBoostRegressor

catb_rmse = []
catb_mae = []
catb_mse = []
catb_r2 = []

# Without feature scaling
catb.fit(X_train, y_train)
pred = catb.predict(X_test)
catb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
catb_mae.append(mean_absolute_error(y_test, pred))
catb_mse.append(mean_squared_error(y_test, pred))
catb_r2.append(r2_score(y_test, pred))

# Apply different scaling techniques and make predictions using CatBoost
for scaler, scaler_name in zip(scalers, scaler_names):
    pipe = make_pipeline(scaler, catb)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    catb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
    catb_mae.append(mean_absolute_error(y_test, pred))
    catb_mse.append(mean_squared_error(y_test, pred))
    catb_r2.append(r2_score(y_test, pred))

# Create a DataFrame to display the results
catb_df = pd.DataFrame({
    'Root Mean Squared Error': catb_rmse,
    'Mean Absolute Error': catb_mae,
    'Mean Squared Error': catb_mse,
    'R-squared': catb_r2
}, index=['Original'] + scaler_names)
catb_df



# ## Linear Regression

# In[46]:


from sklearn.linear_model import LinearRegression

linear_rmse = []
linear_mae = []
linear_mse = []
linear_r2 = []

# Without feature scaling
linear.fit(X_train, y_train)
pred = linear.predict(X_test)
linear_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
linear_mae.append(mean_absolute_error(y_test, pred))
linear_mse.append(mean_squared_error(y_test, pred))
linear_r2.append(r2_score(y_test, pred))

# Apply different scaling techniques and make predictions using Linear Regression
for scaler, scaler_name in zip(scalers, scaler_names):
    pipe = make_pipeline(scaler, linear)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    linear_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
    linear_mae.append(mean_absolute_error(y_test, pred))
    linear_mse.append(mean_squared_error(y_test, pred))
    linear_r2.append(r2_score(y_test, pred))

# Create a DataFrame to display the results
linear_df = pd.DataFrame({
    'Root Mean Squared Error': linear_rmse,
    'Mean Absolute Error': linear_mae,
    'Mean Squared Error': linear_mse,
    'R-squared': linear_r2
}, index=['Original'] + scaler_names)
linear_df


# ##  Stochastic Gradient Descent (SGD) 

# In[47]:


sdg_rmse = []
sdg_mae = []
sdg_mse = []
sdg_r2 = []

# Without feature scaling
sdg.fit(X_train, y_train)
pred = sdg.predict(X_test)
sdg_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
sdg_mae.append(mean_absolute_error(y_test, pred))
sdg_mse.append(mean_squared_error(y_test, pred))
sdg_r2.append(r2_score(y_test, pred))

# Apply different scaling techniques and make predictions using SGDRegressor
for scaler, scaler_name in zip(scalers, scaler_names):
    pipe = make_pipeline(scaler, sdg)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    sdg_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
    sdg_mae.append(mean_absolute_error(y_test, pred))
    sdg_mse.append(mean_squared_error(y_test, pred))
    sdg_r2.append(r2_score(y_test, pred))

# Create a DataFrame to display the results
sdg_df = pd.DataFrame({
    'Root Mean Squared Error': sdg_rmse,
    'Mean Absolute Error': sdg_mae,
    'Mean Squared Error': sdg_mse,
    'R-squared': sdg_r2
}, index=['Original'] + scaler_names)
sdg_df


# ## RandomForest Regression

# In[48]:


rfr_rmse = []
rfr_mae = []
rfr_mse = []
rfr_r2 = []

# Without feature scaling
rfr.fit(X_train, y_train)
pred = rfr.predict(X_test)
rfr_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
rfr_mae.append(mean_absolute_error(y_test, pred))
rfr_mse.append(mean_squared_error(y_test, pred))
rfr_r2.append(r2_score(y_test, pred))

# Apply different scaling techniques and make predictions using RandomForestRegressor
for scaler, scaler_name in zip(scalers, scaler_names):
    pipe = make_pipeline(scaler, rfr)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rfr_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
    rfr_mae.append(mean_absolute_error(y_test, pred))
    rfr_mse.append(mean_squared_error(y_test, pred))
    rfr_r2.append(r2_score(y_test, pred))

# Create a DataFrame to display the results
rfr_df = pd.DataFrame({
    'Root Mean Squared Error': rfr_rmse,
    'Mean Absolute Error': rfr_mae,
    'Mean Squared Error': rfr_mse,
    'R-squared': rfr_r2
}, index=['Original'] + scaler_names)
rfr_df


# ## GradientBoosting Regression

# In[49]:


gb_rmse = []
gb_mae = []
gb_mse = []
gb_r2 = []

# Without feature scaling
gb.fit(X_train, y_train)
pred = gb.predict(X_test)
gb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
gb_mae.append(mean_absolute_error(y_test, pred))
gb_mse.append(mean_squared_error(y_test, pred))
gb_r2.append(r2_score(y_test, pred))

# Apply different scaling techniques and make predictions using GradientBoosting
for scaler in scalers:
    pipe = make_pipeline(scaler, gb)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    gb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
    gb_mae.append(mean_absolute_error(y_test, pred))
    gb_mse.append(mean_squared_error(y_test, pred))
    gb_r2.append(r2_score(y_test, pred))

# Show results
gb_df = pd.DataFrame({
    'Root Mean Squared Error': gb_rmse,
    'Mean Absolute Error': gb_mae,
    'Mean Squared Error': gb_mse,
    'R-squared': gb_r2
}, index=['Original', 'MinMaxScaler', 'StandardScaler'])
gb_df


# ## Bayesian Ridge Regression

# In[50]:


br_rmse = []
br_mae = []
br_mse = []
br_r2 = []

# Without feature scaling
br.fit(X_train, y_train)
pred = br.predict(X_test)
br_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
br_mae.append(mean_absolute_error(y_test, pred))
br_mse.append(mean_squared_error(y_test, pred))
br_r2.append(r2_score(y_test, pred))

# Apply different scaling techniques and make predictions using BayesianRidge
for scaler, scaler_name in zip(scalers, scaler_names):
    pipe = make_pipeline(scaler, br)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    br_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
    br_mae.append(mean_absolute_error(y_test, pred))
    br_mse.append(mean_squared_error(y_test, pred))
    br_r2.append(r2_score(y_test, pred))

# Create a DataFrame to display the results
br_df = pd.DataFrame({
    'Root Mean Squared Error': br_rmse,
    'Mean Absolute Error': br_mae,
    'Mean Squared Error': br_mse,
    'R-squared': br_r2
}, index=['Original'] + scaler_names)
br_df


# In[51]:


data = pd.DataFrame([
    ['KNN', 'MinMaxScaler', knn_rmse[1], knn_mae[1], knn_mse[1], knn_r2[1]],
    ['KNN', 'StandardScaler', knn_rmse[2], knn_mae[2], knn_mse[2], knn_r2[2]],
    ['SVR', 'MinMaxScaler', svr_rmse[1], svr_mae[1], svr_mse[1], svr_r2[1]],
    ['SVR', 'StandardScaler', svr_rmse[2], svr_mae[2], svr_mse[2], svr_r2[2]],
    ['XGBoost', 'MinMaxScaler', xgb_rmse[1], xgb_mae[1], xgb_mse[1], xgb_r2[1]],
    ['XGBoost', 'StandardScaler', xgb_rmse[2], xgb_mae[2], xgb_mse[2], xgb_r2[2]],
    ['CatBoost', 'MinMaxScaler', catb_rmse[1], catb_mae[1], catb_mse[1], catb_r2[1]],
    ['CatBoost', 'StandardScaler', catb_rmse[2], catb_mae[2], catb_mse[2], catb_r2[2]],
    ['Linear', 'MinMaxScaler', linear_rmse[1], linear_mae[1], linear_mse[1], linear_r2[1]],
    ['Linear', 'StandardScaler', linear_rmse[2], linear_mae[2], linear_mse[2], linear_r2[2]],
    ['SGD', 'MinMaxScaler', sdg_rmse[1], sdg_mae[1], sdg_mse[1], sdg_r2[1]],
    ['SGD', 'StandardScaler', sdg_rmse[2], sdg_mae[2], sdg_mse[2], sdg_r2[2]],
    ['RF', 'MinMaxScaler', rfr_rmse[1], rfr_mae[1], rfr_mse[1], rfr_r2[1]],
    ['RF', 'StandardScaler', rfr_rmse[2], rfr_mae[2], rfr_mse[2], rfr_r2[2]],
    ['GB', 'MinMaxScaler', gb_rmse[1], gb_mae[1], gb_mse[1], gb_r2[1]],
    ['GB', 'StandardScaler', gb_rmse[2], gb_mae[2], gb_mse[2], gb_r2[2]],
    ['BR', 'MinMaxScaler', br_rmse[1], br_mae[1], br_mse[1], br_r2[1]],
    ['BR', 'StandardScaler', br_rmse[2], br_mae[2], br_mse[2], br_r2[2]]
], columns=['Models', 'Scalers', 'Root Mean Squared Error', 'Mean Absolute Error', 'Mean Squared Error', 'R-squared'])

data



# In[52]:


fig, ax = plt.subplots(figsize=(10, 6))

models = data['Models'].unique()
scalers = data['Scalers'].unique()
bar_width = 0.35
opacity = 0.8

for i, scaler in enumerate(scalers):
    scaled_data = data[data['Scalers'] == scaler]
    x = np.arange(len(models))
    ax.bar(x + i * bar_width, scaled_data['Root Mean Squared Error'], bar_width, alpha=opacity, label=scaler)

ax.set_title('Root Mean Squared Error for Different Models and Scaling Techniques')
ax.set_xlabel('Models')
ax.set_ylabel('Root Mean Squared Error')
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout()
plt.show()


# # Classification Model

# In[53]:


normalised_features


# In[54]:


# Splitting data into train and test data
#X_train, X_test, y_train, y_test = train_test_split(normalised_features, y, test_size = 0.30, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# In[55]:


# Normal scaling of training dataset
X_train = normal.fit_transform(X_train)  
X_test = normal.transform(X_test)


# # RandomOverSampler is a technique used to address class imbalance in  datasets 
# 
# #### works by randomly duplicating samples from the minority class(es) in order to balance the class distribution.
# 
# #### increases the number of instances in the minority class(es), allowing the model to learn from a more balanced dataset.

# In[56]:


from imblearn.over_sampling import RandomOverSampler
over_sampler = RandomOverSampler(random_state = 42)
X_res, y_res = over_sampler.fit_resample(X_train, y_train)
y_res.value_counts()


# ##### now, X_res ,y_res represent X_train, y_train after data balancing

# ## Logistic Regression 

# In[57]:


#The LogisticRegression class can be configured for multinomial logistic regression 
#by setting the “multi_class” argument to “multinomial” and the “solver” argument to a solver 
#that supports multinomial logistic regression, such as “lbfgs“.
lr = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')
lr.fit(X_train, y_train)
score_lr_without_Kfold = lr.score(X_test, y_test)
print('Accuracy')
score_lr_without_Kfold


# ## Support vector machines (SVM)
# 
# #### *SVM can be used for both classification and regerssion

# In[58]:


# one-vs-one (‘ovo’) is used for multi-class strategy.
svm = SVC(decision_function_shape='ovo')
svm.fit(X_train, y_train)
score_svm_without_Kfold = svm.score(X_test, y_test)
print('Accuracy')
score_svm_without_Kfold


# ## Random Forest

# In[59]:


# n_estimators is a parameter for the number of trees in the forest, which is 40
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
score_rf_without_Kfold = rf.score(X_test, y_test)
print('Accuracy')
score_rf_without_Kfold


# ## K-Nearest Neighbors Classification

# In[60]:


# n_neighbors is a parameter for Number of neighbors, which is 6
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
score_knn_without_Kfold = knn.score(X_test, y_test)
print('Accuracy')
score_knn_without_Kfold


# ## XGBoost Classification

# In[61]:


xgb = XGBClassifier()
xgb.fit(X_train, y_train)
score_xgb_without_Kfold = xgb.score(X_test, y_test)
print('Accuracy')
score_xgb_without_Kfold


# ## Cross Validation techniques -- Evaluate the performance
# ## Using Imbalance data (X_train, y_train)
# #### K-fold as 3

# ## Logistic Regression 

# In[62]:


score_lr_with_Kfold_imbalance = cross_val_score(LogisticRegression(solver='lbfgs',multi_class='multinomial'), 
                                                X_train, y_train, cv=3)
print("Evaluation metric scores for each fold : ",score_lr_with_Kfold_imbalance)
print("Avg :",np.average(score_lr_with_Kfold_imbalance))


# ## Support Vector Machine (SVM) 

# In[63]:


score_svm_with_Kfold_imbalance = cross_val_score(SVC(decision_function_shape='ovo'), X_train, y_train, cv=3)
print("Evaluation metric scores for each fold : ",score_svm_with_Kfold_imbalance)
print("Avg :",np.average(score_svm_with_Kfold_imbalance))


# ## Random Forest Classification 

# In[64]:


score_rf_with_Kfold_imbalance = cross_val_score(RandomForestClassifier(n_estimators=40), X_train, y_train, cv=10)
print("Evaluation metric scores for each fold : ",score_rf_with_Kfold_imbalance)
print("Avg :",np.average(score_rf_with_Kfold_imbalance))


# ## K-Nearest Neighbors Classification

# In[65]:


# Check cross val scores of KNeighborsClassifier with with K-fold as 10.
score_knn_with_Kfold_imbalance = cross_val_score(KNeighborsClassifier(n_neighbors=6), X_train, y_train, cv=10)
print("Evaluation metric scores for each fold : ",score_knn_with_Kfold_imbalance)
print("Avg :",np.average(score_knn_with_Kfold_imbalance))


# # XGBoost Classification

# In[66]:


# Check cross val scores of XGBClassifier with with K-fold as 3.
score_xgb_with_Kfold_imbalance = cross_val_score(XGBClassifier(), X_train, y_train, cv=3)
print("Evaluation metric scores for each fold : ",score_xgb_with_Kfold_imbalance)
print("Avg :",np.average(score_xgb_with_Kfold_imbalance))


# In[67]:


# With imbalance dataset, score of RandomForestClassifier is high 
# hence reverified with differnt estimators but n_estimators=40 gives good score
scores1 = cross_val_score(RandomForestClassifier(n_estimators=5),X_train, y_train, cv=10)
print("Avg Score for Estimators=5 and CV=10 :",np.average(scores1))
scores2 = cross_val_score(RandomForestClassifier(n_estimators=10), X_train, y_train, cv=10)
print("Avg Score for Estimators=10 and CV=10 :",np.average(scores1))
scores3 = cross_val_score(RandomForestClassifier(n_estimators=20),X_train, y_train, cv=10)
print("Avg Score for Estimators=20 and CV=10 :",np.average(scores1))
scores4 = cross_val_score(RandomForestClassifier(n_estimators=30), X_train, y_train, cv=10)
print("Avg Score for Estimators=30 and CV=10 :",np.average(scores1))


# # Using Balanced Data

# In[68]:


# cross validation scores with balance dataset
score_lr_with_Kfold_balance = cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X_res, y_res, cv=3)
print(score_lr_with_Kfold_balance)
print("Avg :",np.average(score_lr_with_Kfold_balance))
score_svm_with_Kfold_balance = cross_val_score(SVC(gamma='auto'), X_res, y_res, cv=3)
print(score_svm_with_Kfold_balance)
print("Avg :",np.average(score_svm_with_Kfold_balance))
score_rf_with_Kfold_balance = cross_val_score(RandomForestClassifier(n_estimators=40),X_res, y_res, cv=10)
print(score_rf_with_Kfold_balance)
print("Avg :",np.average(score_rf_with_Kfold_balance))
score_knn_with_Kfold_balance = cross_val_score(KNeighborsClassifier(n_neighbors=6), X_res, y_res, cv=10)
print(score_knn_with_Kfold_balance)
print("Avg :",np.average(score_knn_with_Kfold_balance))
score_xgb_with_Kfold_balance = cross_val_score(XGBClassifier(), X_res, y_res, cv=10)
print(score_xgb_with_Kfold_balance)
print("Avg :",np.average(score_xgb_with_Kfold_balance))


# In[69]:


import matplotlib.pyplot as plt
import numpy as np

models = ['Logistic Regression', 'SVM', 'Random Forest', 'KNN', 'XGBoost']
scores = [
    np.average(score_lr_with_Kfold_balance),
    np.average(score_svm_with_Kfold_balance),
    np.average(score_rf_with_Kfold_balance),
    np.average(score_knn_with_Kfold_balance),
    np.average(score_xgb_with_Kfold_balance)
]

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(models, scores)
plt.xlabel('Models')
plt.ylabel('Average Score')
plt.title('Cross-Validation Scores with Balanced Dataset')
plt.ylim([0, 1])  # Set the y-axis limit between 0 and 1
plt.grid(True)
plt.show()


# In[70]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tabulate import tabulate

n_estimators = [5, 10, 20, 30, 40]
cv_folds = 10

scores = []

for estimator in n_estimators:
    clf = RandomForestClassifier(n_estimators=estimator)
    cv_scores = cross_val_score(clf, X_res, y_res, cv=cv_folds)
    avg_score = np.average(cv_scores)
    scores.append(avg_score)

# Create a table using tabulate
table = zip(n_estimators, scores)
headers = ["n_estimators", "Average Score"]
table_str = tabulate(table, headers, tablefmt="fancy_grid")

# Print the table
print ("CV = 10")
print(table_str)


# In[71]:


import matplotlib.pyplot as plt

n_estimators = [5, 10, 20, 30, 40]
cv_folds = 10

scores = []

for estimator in n_estimators:
    clf = RandomForestClassifier(n_estimators=estimator)
    cv_scores = cross_val_score(clf, X_res, y_res, cv=cv_folds)
    avg_score = np.average(cv_scores)
    scores.append(avg_score)

# Plot the line chart
plt.plot(n_estimators, scores, marker='o')
plt.xlabel('n_estimators')
plt.ylabel('Average Score')
plt.title('Average Score vs. n_estimators')
plt.grid(True)
plt.show()


# In[72]:


# With balance dataset, score of RandomForestClassifier is high 
# hence reverified with differnt estimators but n_estimators=40 gives good score
scores1 = cross_val_score(RandomForestClassifier(n_estimators=5),X_res, y_res, cv=10)
print("Avg Score for Estimators=5 and CV=10 :",np.average(scores1))
scores2 = cross_val_score(RandomForestClassifier(n_estimators=10),X_res, y_res, cv=10)
print("Avg Score for Estimators=10 and CV=10 :",np.average(scores1))
scores3 = cross_val_score(RandomForestClassifier(n_estimators=20),X_res, y_res, cv=10)
print("Avg Score for Estimators=20 and CV=10 :",np.average(scores1))
scores4 = cross_val_score(RandomForestClassifier(n_estimators=30),X_res, y_res, cv=10)
print("Avg Score for Estimators=30 and CV=10 :",np.average(scores1))


# In[73]:


# Bar subplots for checking differnce between original, k-folded imbalanced and k-folded balanced data for differnt models
data = pd.DataFrame([['LogisticRegression', 'without_Kfold', score_lr_without_Kfold], 
                   ['LogisticRegression', 'with_Kfold_imbalance', score_lr_with_Kfold_imbalance], 
                   ['LogisticRegression', 'with_Kfold_balance', score_lr_with_Kfold_balance], 
                   ['SVM', 'without_Kfold', score_svm_without_Kfold], 
                   ['SVM', 'with_Kfold_imbalance', score_svm_with_Kfold_imbalance], 
                   ['SVM', 'with_Kfold_balance', score_svm_with_Kfold_balance],
                   ['RandomForest', 'without_Kfold', score_rf_without_Kfold], 
                   ['RandomForest', 'with_Kfold_imbalance', score_rf_with_Kfold_imbalance], 
                   ['RandomForest', 'with_Kfold_balance', score_rf_with_Kfold_balance],
                   ['KNN', 'without_Kfold', score_knn_without_Kfold], 
                   ['KNN', 'with_Kfold_imbalance', score_knn_with_Kfold_imbalance], 
                   ['KNN', 'with_Kfold_balance', score_knn_with_Kfold_balance],
                   ['XGBoost', 'without_Kfold', score_xgb_without_Kfold], 
                   ['XGBoost', 'with_Kfold_imbalance', score_xgb_with_Kfold_imbalance], 
                   ['XGBoost', 'with_Kfold_balance', score_xgb_with_Kfold_balance]], 
                  columns=['Models', 'Processes', 'Cross Validation Scores'])

data = data.explode('Cross Validation Scores')
data['Cross Validation Scores'] = data['Cross Validation Scores'].astype('float') * 100

# plot with seaborn barplot
plt.figure(figsize=(18, 8))
ax = sns.barplot(data=data, x ='Processes', y ='Cross Validation Scores', hue ='Models', ci = None)

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3)


# # Model prediction and evaluation
# 

# ## RandomForestClassifier
# 

# In[74]:


# Fitting balanced data into RandomForestClassifier
RF = RandomForestClassifier(criterion='gini')
RF.fit(X_res, y_res)
# Predicting unseen data with RandomForestClassifier
pred= RF.predict(X_test)
# Check mean_squared_error
print(np.sqrt(mean_squared_error(y_test, pred)))


# In[75]:


# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))


# ## KNeighborsClassifier

# In[76]:


# Fitting balanced data into KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_res, y_res)
# Predicting unseen data with KNeighborsClassifier
pred= knn.predict(X_test)
# Check mean_squared_error
print(np.sqrt(mean_squared_error(y_test, pred)))


# In[77]:


# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))


# The classification report indicates relatively high precision, recall, and F1-scores for each class, suggesting good accuracy and effectiveness in classifying the data. The overall accuracy of the model is 79%, meaning it correctly predicted the class labels for approximately 79% of the samples in the test set

# # XGBoost Classifier

# In[78]:


xgb = XGBClassifier()
xgb.fit(X_res, y_res)
# Predicting unseen data with XGBClassifier
pred= xgb.predict(X_test)
# Check mean_squared_error
print(np.sqrt(mean_squared_error(y_test, pred)))


# In[79]:


# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))


# The weighted average F1-score is 0.78, indicating the overall effectiveness of the model in classification. The accuracy of the model is 78%, meaning it correctly predicted the class labels for approximately 78% of the samples in the test set.
# 
# In summary, the model shows reasonably good performance, with high accuracy, and relatively high precision, recall, and F1-scores for each class. However, there is room for improvement, particularly in correctly identifying instances of class 1.

# # DecisionTree Classifier

# In[80]:


# Fitting balanced data into DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_res, y_res)
# Predicting unseen data with DecisionTreeClassifier
pred= model.predict(X_test)
# Check mean_squared_error
print(np.sqrt(mean_squared_error(y_test, pred)))


# In[81]:


# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))


# The weighted average F1-score is 0.81, indicating the overall effectiveness of the model in classification. The accuracy of the model is 81%, meaning it correctly predicted the class labels for approximately 81% of the samples in the test set.
# 
# In summary, the model shows good performance, with high accuracy and relatively high precision, recall, and F1-scores for each class. However, there is still room for improvement, particularly in correctly identifying instances of class 1.

# # Support Vector Machines 

# In[82]:


# Fitting balanced data into SVC
svm = SVC(decision_function_shape='ovo')
svm.fit(X_res, y_res)
# Predicting unseen data with SVC
pred = svm.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))


# In[83]:


# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))


# The weighted average F1-score is 0.66, indicating the overall effectiveness of the model in classification. The accuracy of the model is 68%, meaning it correctly predicted the class labels for approximately 68% of the samples in the test set.
# 
# In summary, the model shows mixed performance. It performs well in identifying class 0 and class 2 but struggles with class 1. The overall accuracy and F1-score are relatively low, suggesting room for improvement in the model's performance.

# ## Radial basis function kernel

# In[84]:


# Fitting balanced data into SVM RBF
svm_rbf = SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo')
svm_rbf.fit(X_res, y_res)
# Predicting unseen data with SVM RBF
pred = svm_rbf.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))


# In[85]:


# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))


# The weighted average F1-score is 0.65, indicating the overall effectiveness of the model in classification. The accuracy of the model is 67%, meaning it correctly predicted the class labels for approximately 67% of the samples in the test set.
# 
# In summary, the model shows mixed performance. It performs well in identifying class 0 and class 2 but struggles with class 1. The overall accuracy and F1-score are relatively low, suggesting room for improvement in the model's performance.

# # Gaussian Naive Bayes

# In[86]:


from sklearn.naive_bayes import GaussianNB
# Fitting balanced data into GaussianNB
gnb = GaussianNB()
gnb.fit(X_res, y_res)
# Predicting unseen data with GaussianNB
pred = gnb.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))


# In[87]:


# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))


# The weighted average F1-score is 0.56, indicating the overall effectiveness of the model in classification. The accuracy of the model is 60%, meaning it correctly predicted the class labels for approximately 60% of the samples in the test set.
# 
# In summary, the model shows mixed performance. It performs reasonably well in identifying class 0 and class 2, but struggles with class 1. The overall accuracy and F1-score are relatively low, suggesting room for improvement in the model's performance

# # Logistic Regression

# In[88]:


# Fitting balanced data into LogisticRegression
lr = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')
lr.fit(X_res, y_res)
# Predicting unseen data with LogisticRegression
pred = lr.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))


# In[89]:


# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))


# The weighted average F1-score is 0.61, indicating the overall effectiveness of the model in classification. The accuracy of the model is 61%, meaning it correctly predicted the class labels for approximately 61% of the samples in the test set.
# 
# In summary, the model shows mixed performance. It performs reasonably well in identifying class 0 and class 2, but struggles with class 1. The overall accuracy and F1-score are moderate, suggesting some room for improvement in the model's performance.

# # Hyper-parameters tuning of an estimator

# # XGBClassifier is choosen
# # KNeighborsClassifier is choosen
# 

# In[90]:


from sklearn.model_selection import GridSearchCV
# Initialising list of paramaters for selection of best params for XGBoost Model
param_grid = {
    "learning_rate": [0.5, 1, 3, 5],
    "reg_lambda": [0, 1, 5, 10, 20]
}


# In[91]:


# Applying param_grid , k_fold as 3 and training the model
# Computations can be run in parallel by using the keyword n_jobs=-1
grid = GridSearchCV(xgb, param_grid, cv=3, n_jobs=-1)
grid.fit(X_res, y_res)


# In[92]:


grid.best_params_


# In[93]:


# Applying Best params to XGBoost Model
xgb = XGBClassifier(colsample_bytree= 1, gamma=0, learning_rate=1, max_depth=3, subsample=0.8, reg_lambda=1)
xgb.fit(X_res, y_res)
pred= xgb.predict(X_test)


# In[94]:


# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))


# In[95]:


# Initialising list of paramaters for selection of best params for KNeighborsClassifier Model
# Applying param_grid and training the model
param_grid={'n_neighbors': [1,2,3,4,5,6,7,8]}
gridsearchcv = GridSearchCV(knn, param_grid)
gridsearchcv.fit(X_res, y_res)


# In[96]:


gridsearchcv.best_params_


# In[97]:


# Applying Best Params to KNeighborsClassifier Model
knn2 = KNeighborsClassifier(n_neighbors=1)
knn2.fit(X_res, y_res)
pred = knn2.predict(X_test)


# In[98]:


cm = confusion_matrix(y_test, pred)
print("Confusion Matrix:")
print(cm)


# In[99]:


report = classification_report(y_test, pred)
print("Classification Report:")
print(report)


# # Model Decision : XGBoost Classifier
# I chose the XGBoost classifier as my model due to its strong performance demonstrated in the confusion matrix and classification report. The initial evaluation of the model yielded a confusion matrix with precision, recall, and F1-scores ranging from 0.70 to 0.84 across the three classes. These metrics indicate a reasonably accurate and balanced classification performance for the model. The weighted average F1-score of 0.78 and overall accuracy of 0.78 further support the effectiveness of the model in capturing the underlying patterns in the data.
# 
# After tuning the XGBoost classifier, the model's performance improved slightly. The updated confusion matrix shows a consistent pattern of precision, recall, and F1-scores across the classes, with minimal changes compared to the initial evaluation. The weighted average F1-score remained at 0.79, indicating the model's consistency in capturing the overall predictive performance. The accuracy of 0.79 suggests that the model correctly predicted the class labels for approximately 79% of the samples in the test set.
# 
# These results demonstrate that the XGBoost classifier is a suitable choice for the classification task at hand. It exhibits consistent and reliable performance, maintaining high precision, recall, and F1-scores across the different classes. Furthermore, the model's ability to maintain its performance after hyperparameter tuning highlights its robustness and suitability for the dataset. Overall, the XGBoost classifier is a strong candidate for accurately classifying the data and can be relied upon for further analysis and decision-making.

# In[100]:


import pandas as pd
data = pd.read_csv('Maternal Health Risk Data Set.csv')

X = data.drop('RiskLevel', axis = 1)
Y = data['RiskLevel']


# In[101]:


import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
fn = data.columns[0:6]
cn = data["RiskLevel"].unique().tolist()
dataTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
dataTree.fit(X,Y)
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize= (20,10), dpi=300)
tree.plot_tree(dataTree, feature_names= fn, class_names= cn, filled = True)
plt.show()


# In[102]:


data.drop(columns="SystolicBP", axis=1, inplace=True)
data


# In[103]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# In[104]:


normal = MinMaxScaler()  
X_train_features = normal.fit_transform(X_train)  
X_test_features = normal.transform(X_test)


# In[105]:


over_sampler = RandomOverSampler(random_state = 42)
X_res, y_res = over_sampler.fit_resample(X_train, y_train)
y_res.value_counts()


# In[106]:


RF2= RandomForestClassifier(criterion='gini', max_depth=20, max_features='log2', n_estimators=50)
RF2.fit(X_res, y_res)
pred= RF2.predict(X_test)


# In[107]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(y_test, pred)

# Create a heatmap plot of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[108]:


from sklearn.metrics import classification_report

report = classification_report(y_test, pred)
print("Classification Report:")
print(report)


# # Prediction application (demostration)

# In[ ]:


# Get user input for the new data point
age = int(input("Enter Age: "))
systolicBP = int(input("Enter Systolic BP: "))
diastolicBP = int(input("Enter Diastolic BP: "))
bloodSugar = float(input("Enter BS: "))
bodyTemp = float(input("Enter Body Temperature(in C): "))
heartRate = int(input("Enter Heart Rate: "))

# Create a new data point based on user inputs
new_data = [[age, systolicBP, diastolicBP, bloodSugar, bodyTemp, heartRate]]

# Select the appropriate scaler object based on your choice (e.g., normal or standard)
scaler = normal

# Scale the new data using the selected scaler object
new_data_scaled = scaler.transform(new_data)

# Make a prediction using the tuned XGBoost classifier
prediction = xgb.predict(new_data_scaled)

# Print the prediction
print("Predicted Risk Level:", prediction)






















