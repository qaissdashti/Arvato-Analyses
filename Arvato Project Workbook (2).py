#!/usr/bin/env python
# coding: utf-8

# # Capstone Project: Create a Customer Segmentation Report for Arvato Financial Services
# 
# In this project, you will analyze demographics data for customers of a mail-order sales company in Germany, comparing it against demographics information for the general population. You'll use unsupervised learning techniques to perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company. Then, you'll apply what you've learned on a third dataset with demographics information for targets of a marketing campaign for the company, and use a model to predict which individuals are most likely to convert into becoming customers for the company. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# If you completed the first term of this program, you will be familiar with the first part of this project, from the unsupervised learning project. The versions of those two datasets used in this project will include many more features and has not been pre-cleaned. You are also free to choose whatever approach you'd like to analyzing the data rather than follow pre-determined steps. In your work on this project, make sure that you carefully document your steps and decisions, since your main deliverable for this project will be a blog post reporting your findings.

# In[1]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score


# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# ## Part 0: Get to Know the Data
# 
# There are four data files associated with this project:
# 
# - `Udacity_AZDIAS_052018.csv`: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
# - `Udacity_CUSTOMERS_052018.csv`: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
# - `Udacity_MAILOUT_052018_TRAIN.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
# - `Udacity_MAILOUT_052018_TEST.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. Use the information from the first two files to figure out how customers ("CUSTOMERS") are similar to or differ from the general population at large ("AZDIAS"), then use your analysis to make predictions on the other two files ("MAILOUT"), predicting which recipients are most likely to become a customer for the mail-order company.
# 
# The "CUSTOMERS" file contains three extra columns ('CUSTOMER_GROUP', 'ONLINE_PURCHASE', and 'PRODUCT_GROUP'), which provide broad information about the customers depicted in the file. The original "MAILOUT" file included one additional column, "RESPONSE", which indicated whether or not each recipient became a customer of the company. For the "TRAIN" subset, this column has been retained, but in the "TEST" subset it has been removed; it is against that withheld column that your final predictions will be assessed in the Kaggle competition.
# 
# Otherwise, all of the remaining columns are the same between the three data files. For more information about the columns depicted in the files, you can refer to two Excel spreadsheets provided in the workspace. [One of them](./DIAS Information Levels - Attributes 2017.xlsx) is a top-level list of attributes and descriptions, organized by informational category. [The other](./DIAS Attributes - Values 2017.xlsx) is a detailed mapping of data values for each feature in alphabetical order.
# 
# In the below cell, we've provided some initial code to load in the first two datasets. Note for all of the `.csv` data files in this project that they're semicolon (`;`) delimited, so an additional argument in the [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call has been included to read in the data properly. Also, considering the size of the datasets, it may take some time for them to load completely.
# 
# You'll notice when the data is loaded in that a warning message will immediately pop up. Before you really start digging into the modeling and analysis, you're going to need to perform some cleaning. Take some time to browse the structure of the data and look over the informational spreadsheets to understand the data values. Make some decisions on which features to keep, which features to drop, and if any revisions need to be made on data formats. It'll be a good idea to create a function with pre-processing steps, since you'll need to clean all of the datasets before you work with them.

# In[2]:


# load in the data

azdias = pd.read_csv('../../data/Term2/capstone/arvato_data/Udacity_AZDIAS_052018.csv', sep=';')
customers = pd.read_csv('../../data/Term2/capstone/arvato_data/Udacity_CUSTOMERS_052018.csv', sep=';')


# In[3]:


azdias.shape


# In[4]:


customers.shape


# In[7]:


sns.countplot(azdias.isnull().sum())


# In[8]:


sns.countplot(customers.isnull().sum())


# In[ ]:





# In[ ]:





# ## Part 1: Customer Segmentation Report
# 
# The main bulk of your analysis will come in this part of the project. Here, you should use unsupervised learning techniques to describe the relationship between the demographics of the company's existing customers and the general population of Germany. By the end of this part, you should be able to describe parts of the general population that are more likely to be part of the mail-order company's main customer base, and which parts of the general population are less so.

# <b> Steps to be taken at preprocessing stage:</b>
# 
# Excel file cleanup:
# 1. Read in the excel file
# 2. Fill forward the empty lines with the line above for better element retrieval
# 3. Retrieve the missing and or the unknown labels and place in list
# 4. Split the missing and unknown labels and loop through them to get the unique elements only
# 5. Concat the attribute names with a list of the unique elements
# 
# 
# Within the processing function the below will take place for both files of azdias and customer:
# 1. All ints must be converted to floats
# 2. Check is row in the file with the same attribute in the excel file, if available the retrieve the element if not then NAN
# 3. Use separate function to choose the cutoff level, then place the cols to be dropped in the preprocessing function. columns already placed in function
# 4. choose the number of row needed to be removed as a custoff, also can be calculated in choose_cut_off_row function.
# 5. All changes are in place and a df is returned
# 
# Outside the preprocessing function step:
# 1. to remove the categorical cols
# 2. to test out using both functions "choose_cut_off" and "choose_cut_off_row" to choose the cols and rows that need be removed
# 3. the three cols PRODUCT_GROUP CUSTOMER_GROUP ONLINE_PURCHASE, must be removed separately
# 
# Note: to ensure that both files have the similar cols names and same number of features before moving further.
# 
# 
# 

# In[94]:


customers.shape


# In[38]:


azdias.shape


# In[91]:


# convert all dtypes of int to float, and keep the object type - testing part
for col in azdias.columns:
    if azdias[col].dtype == np.int64:
        azdias[col] = azdias[col].astype(np.float64)
    
#check if all azdias dtypes are floats and objects
azdias.dtypes.value_counts()


# <b> We start with the excel cleanup</b>
# 
# <b>Convert and correct the excel file to ensure it can be used to clean and pull the needed attributes from for the azdias file</b>
#     
# first step to clean the excel sheet with the attributes to be used
# for cleaning the general population file

# In[3]:


#first step to clean the excel sheet with the attributes to be used
# for cleaning the general population file

#read in the file
excel_att = pd.read_excel('DIAS Attributes - Values 2017.xlsx')
del excel_att['Unnamed: 0']
excel_att_n = excel_att['Attribute'].fillna(method='ffill')
excel_att['Attribute'] = excel_att_n
excel_att_table = excel_att[(excel_att['Meaning'].str.contains('unknown')) | (excel_att['Meaning'].str.contains('no'))]
#excel_att_table.head()

unknowns = []

for att in excel_att_table['Attribute'].unique():
    _ = excel_att_table.loc[excel_att_table['Attribute'] == att, 'Value'].astype(str).str.cat(sep=',')
    _ = _.split(',')
    unknowns.append(_)
    
#unknowns

excell_attibutes_com = pd.concat([ pd.Series(excel_att_table['Attribute'].unique()) , pd.Series(unknowns)], axis=1)
excell_attibutes_com.columns= ['attribute', 'missing_or_unknown']
excell_attibutes_com.head()


# <b>Loop to read in the labels or add NaNs</b>
# 
# Below is the loop to clean the azdias data by using the excel sheet provided and add a np.nan inplace of missing values that are being read from the excell_attibutes_com file
# 

# In[114]:


#testing part
for row in excell_attibutes_com['attribute']:
    print(row)
    if row in azdias.columns:
        na_map = excell_attibutes_com.loc[excell_attibutes_com['attribute'] == row, 'missing_or_unknown'].values[0]
        na_idx = azdias.loc[:, row].isin(na_map)
        azdias.loc[na_idx, row] = np.nan
    else:
        continue


# In[16]:


def preprocess_df(df, att_df):
    #function to clean the df and retreive the required labels to
    # convert the unkowns to correct NaN
    
    #first to convert all dtypes to float and keep object type same
    for col in df.columns:
        if df[col].dtype == np.int64:
            df[col] = df[col].astype(np.float64)
            
    #to retrieve the labels (unkown) and change to nan
    for row in att_df['attribute']:
        if row in df.columns:
            nan_list = att_df.loc[att_df['attribute'] == row, 'missing_or_unknown'].values[0]
            nan_index = df.loc[:, row].isin(nan_list)
            df.loc[nan_index, row] = np.nan
        else:
            continue
            
    columns_to_drop = ['AGER_TYP', 'ALTER_HH', 'ALTER_KIND1', 'ALTER_KIND2', 'ALTER_KIND3',
       'ALTER_KIND4', 'D19_BANKEN_ANZ_12', 'D19_BANKEN_ANZ_24',
       'D19_BANKEN_DATUM', 'D19_BANKEN_OFFLINE_DATUM',
       'D19_BANKEN_ONLINE_DATUM', 'D19_BANKEN_ONLINE_QUOTE_12',
       'D19_GESAMT_ANZ_12', 'D19_GESAMT_ANZ_24', 'D19_GESAMT_DATUM',
       'D19_GESAMT_OFFLINE_DATUM', 'D19_GESAMT_ONLINE_DATUM',
       'D19_GESAMT_ONLINE_QUOTE_12', 'D19_TELKO_ANZ_12', 'D19_TELKO_ANZ_24',
       'D19_TELKO_DATUM', 'D19_TELKO_OFFLINE_DATUM', 'D19_TELKO_ONLINE_DATUM',
       'D19_VERSAND_ANZ_12', 'D19_VERSAND_ANZ_24', 'D19_VERSAND_DATUM',
       'D19_VERSAND_OFFLINE_DATUM', 'D19_VERSAND_ONLINE_DATUM',
       'D19_VERSAND_ONLINE_QUOTE_12', 'D19_VERSI_ANZ_12', 'D19_VERSI_ANZ_24',
       'EXTSEL992', 'KBA05_ANHANG', 'KBA05_ANTG1', 'KBA05_ANTG2',
       'KBA05_ANTG3', 'KBA05_ANTG4', 'KBA05_BAUMAX', 'KBA05_CCM4', 'KBA05_KW3',
       'KBA05_MAXVORB', 'KBA05_MOD1', 'KBA05_MOD8', 'KBA05_MOTRAD',
       'KBA05_SEG1', 'KBA05_SEG5', 'KBA05_SEG6', 'KBA05_SEG7', 'KBA05_SEG8',
       'KBA05_SEG9', 'KK_KUNDENTYP', 'PLZ8_ANTG4', 'TITEL_KZ']
    
    #cutt_off_30 = df.columns[df.isnull().sum() / df.shape[0] > percentage]           
    
    #columns to drop based on theshold chosen
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    #row to be dropbed based on the number argument
    row_to_drop = df.index[df.isnull().sum(axis=1) > 35]
    
    df.drop(row_to_drop, axis=0, inplace=True)
    
    

    return df


# In[7]:


get_ipython().run_cell_magic('time', '', 'azdias_pre = preprocess_df(azdias, excell_attibutes_com)')


# In[8]:


get_ipython().run_cell_magic('time', '', 'customers_pre = preprocess_df(customers, excell_attibutes_com)')


# In[9]:


azdias_pre.shape


# In[10]:


customers_pre.shape


# Seperate the 3 additional cols of customer file

# In[11]:


three_col_df_customer = customers_pre[['PRODUCT_GROUP', 'CUSTOMER_GROUP', 'ONLINE_PURCHASE']]
three_col_df_customer.head()


# In[18]:


customers_pre.shape


# In[12]:


#for customer file only

def clean_customer_after_pre(cust_df):
    del cust_df['PRODUCT_GROUP']
    del cust_df['CUSTOMER_GROUP']
    del cust_df['ONLINE_PURCHASE']
    
    customers_pre_no_cat = remove_cat(cust_df)
    return customers_pre_no_cat    


# <b>Categorical columns</b> will be removed due to their labels, and that most have high number of different labels

# In[13]:


#categoricals columns  must be removed manualy!! for both Azdias and Customer file

def remove_cat(df):
    df.columns[df.dtypes == 'object']
    catigorical_col = df.columns[df.dtypes == 'object']
    df.drop(catigorical_col, axis=1, inplace=True)
    return df


# In[14]:


customers_pre_no_cat = clean_customer_after_pre(customers_pre)


# In[15]:


customers_pre_no_cat.shape


# In[16]:


azdias_pre_no_cat = remove_cat(azdias_pre)


# In[17]:


azdias_pre_no_cat.shape


# In[ ]:


# to better analyse and decide on the cat columns and see if to keep or remove entirly. 
for col in azdias.columns[azdias.dtypes == 'object']:
    print(col)
    print(azdias[col].unique())
    


# <b>Removing NaNs based on a cuttoff ratio</b>
# 
# The below two function are used to choose the percentage and number of NaNs elements to be removed - also for testing before using the preprocessing function
# 

# In[86]:


def choose_cut_off(df, percentage):
    #returns the cols that will be removed based on the % number placed in percentage
    cutt_off_30 = df.columns[df.isnull().sum() / df.shape[0] > percentage]           
    
    #columns to drop based on theshold chosen
    df.drop(cutt_off_30, axis=1, inplace=True)


# In[ ]:


def choose_cut_off_row(df, number):
    row_to_drop = df.index[df.isnull().sum(axis=1) > number]
    
    df.drop(row_to_drop, axis=0, inplace=True)
    


# <b>Pickling the files for better space managemnet</b>
# 
# Import and use pickle, to save and serlize the files for better space resources. After cleaning the files we will pickle them so they can be downloaded fast and to be used right away.

# In[124]:


#to upload cleaned files with pickle

#pickle.dump(azdias, open("azdias_clean.pickle", "wb"))

# Dump the customers dataframe to a pickle object to use for later.
#pickle.dump(customers, open("customers_clean.pickle", "wb"))


# In[2]:


# to reload uploaded files back 
#azdias_clean = pickle.load(open("azdias_clean.pickle", "rb"))
#customers_clean = pickle.load(open("customers_clean.pickle", "rb"))


# In[3]:


azdias_clean.shape


# In[4]:


customers_clean.shape


# <b> Assert test</b>
# Test to see if both files have the same number of features before PCA

# In[18]:


def test_df(azdias_df, cust_df):
    assert np.all(cust_df.columns) == np.all(azdias_df.columns)
    assert sum(cust_df.columns == azdias_df.columns) ==307
    print('both files are ready for PCA, with {} columns'.format(len(customers_pre_no_cat.columns)))


# In[19]:


test_df(azdias_pre_no_cat, customers_pre_no_cat)


# <b>Imputing the Nans</b>
# After cleaning and removing the large number of NaNs in the cols and row, we now impute the remaining nans with the most frequent element in the column.

# In[20]:


def impute(az_df, cust_df):
    
# imputer_test = Imputer(strategy='mean or most_frequent', axis=0) to impute col wise
#test_df_1 = pd.DataFrame(imputer_test.fit_transform(test_df_1))
    imputer = Imputer(strategy='most_frequent', axis=0)

    azdias_clean = pd.DataFrame(imputer.fit_transform(az_df),
                                columns=az_df.columns)

    customers_clean = pd.DataFrame(imputer.fit_transform(cust_df),
                                   columns=cust_df.columns)
    
    return azdias_clean, customers_clean


# In[21]:


azdias_clean_pca_ready, customer_clean_pca_ready = impute(azdias_pre_no_cat, customers_pre_no_cat)


# In[22]:


azdias_clean_pca_ready.shape


# In[23]:


customer_clean_pca_ready.shape


# In[24]:


customer_clean_pca_ready.head()


# In[25]:


azdias_clean_pca_ready.head()


# <b>Standardizing</b>
# 
# Before going ahead with the PCA, we need to ensure that both files are standardized and we will use the StandardScaler object here
# 

# In[26]:


scaler = StandardScaler()
azdias_clean = pd.DataFrame(scaler.fit_transform(azdias_clean_pca_ready), columns = azdias_clean_pca_ready.columns)
customers_clean = pd.DataFrame(scaler.transform(customer_clean_pca_ready), columns = customer_clean_pca_ready.columns)


# In[27]:


azdias_clean.head()


# In[28]:


customers_clean.head()


# <b>Files are now ready for PCA and KMEANS:</b>
# 
# Lets Pickle the files so, when we need to retive them we can do so without running all the past functions that take time.

# In[37]:


# Dump the azdias dataframe to a pickle object since it takes up so much room in memory.
pickle.dump(azdias_clean, open("azdias_final.pickle", "wb"))

# Dump the customers dataframe to a pickle object to use for later.
pickle.dump(customers_clean, open("customers_final.pickle", "wb"))


# In[4]:


# Reload cleaned azdias object as saved after above analysis (may need to rerun imports)
azdias_clean = pickle.load(open("azdias_final.pickle", "rb"))

# Reload cleaned customers object as saved after above analysis
customers_clean = pickle.load(open("customers_final.pickle", "rb"))


# In[5]:


azdias_clean.shape


# In[6]:


customers_clean.shape


# <b>Dimensionality Reduction on the Data - PCA</b>
# 
# PCA here will help in reducing the feature space from 307 to 200. After plotting the new features of 200, we can calculate the sum of explained variance which comes to a total of 94% which is high and cover most of the data, and at least reduced it by 93 features

# In[29]:


pca = PCA(200)


# In[30]:


get_ipython().run_cell_magic('time', '', '\nazdias_pca_200 = pca.fit_transform(azdias_clean)')


# In[31]:


azdias_pca_200.shape


# In[32]:


sum(pca.explained_variance_ratio_)


# In[33]:


len(pca.components_)


# In[34]:


pca.components_.shape


# The function below will take in the PCA object and plot the new number of components and how much explained variance they have. Our 200 newly formed features now have a explained variance of about 90%.
# 
# I have chosen 200 features after running the PCA with all the features and the plot showed that at about 200 the explained variance is above 85%.
# 
# Based on that i have decided to go with 200 components.
# 

# In[8]:


def com_plot(pca):
    
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Explained Variance Per Principal Component')


# In[9]:


com_plot(pca)


# In[35]:


pca.components_


# In[36]:


components_200 = pca.components_


# In[37]:


components_200_df = pd.DataFrame(components_200, columns=azdias_clean.columns)
components_200_df.head()


# In[38]:


#list comprehension to create the index
comp = ['comp_' + str(i)for i in components_200_df.index + 1]
comp


# In[39]:


components_200_df.shape


# At this stage i will create a DataFrame from the returned PCA object, after creating the DF then we can visualize the weights of each component and see the correlations between the features and if the provide any insight

# In[40]:


components_200_df.index = comp


# In[41]:


components_200_df.head()


# In[42]:


feature_names_comp = components_200_df.columns
feature_names_comp


# after completing the components DF we will create a function to visualize the weights and see how which features correlate

# <b> Explaining each component </b> and what are the attributes within, and how do they correlate or not , together.

# In[43]:


def display_comp2(components, component_number, weight_n):
    #get the component number for the row
    row_idx = component_number
    #get the values of all features by row
    component_row = components.iloc[row_idx, :].values
    #create a df with both row values and features
    comp_df = pd.DataFrame(list(zip(component_row, components_200_df.columns)),
                          columns=['weights', 'features'])
    #create a column of abs values to retrive later the features
    comp_df['abs'] = comp_df['weights'].apply(lambda x: np.abs(x))
    #sort the values based on weaights
    sorted_df = comp_df.sort_values(by='abs', ascending=False).head(weight_n)
    
    ax = plt.subplots(figsize=(10,6))
    ax = sns.barplot(data=sorted_df,
                    x='weights',
                    y='features',
                    palette='Blues_d')
    ax.set_title("PCA Component Makeup, Component #" + str(component_number))
    plt.show()
    print (sorted_df)
    
    


# In[44]:


display_comp2(components_200_df, 0, 10)


# The 1st component is related to homes with low households 1 to 2 (PLZ8_ANTG1)
# and not correlated to most common type of homes (PLZ8_BAUMAX) but correlated 
# with share of cars per household (KBA13_AUTOQUOTE)
# 

# In[50]:


display_comp2(components_200_df, 1, 10)


# its seems that this component has many things to do with high end expensive cars that are related with each other like BMW, MERC, and sport car
# but are not correlated with cars that have 5 seats(KBA13_SITZE_5)

# In[24]:


display_comp2(components_200_df, 2, 10)


# here the money saver (FINANZ_SPARER) and the unremarkable are highly
# correlated (FINANZ_UNAUFFAELLIGER) and opposite them is the prepared(FINANZ_VORSORGER)

# In[25]:


display_comp2(components_200_df, 3, 10)


# <b> KMEANS and clustering our PCA data</b>
# 
# Before staring a K means cluster, we need to find out what is the best cluster number and to do that we will perform the elbow method, and by plotting the results of the SSE, we can then choose the least SSE that makes sense to then cluster the kmeans with.
# 

# In[21]:


ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    print(k)
    
    # Fit model to samples
    model.fit(PCA_components)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# 
# After plotting, we can see that there is no clear elbow cutoff to choose from but we can see that after number 7 things start becoming linear. We can choose 7 cluster and test it on the data.
# 
# After choosing the number 7, we can fit the data now and perform clustering on the general population data
# 
# 

# We start by creating a Kmeans object from sklearn, then choose the 7 cluster based on the plot above. We then fit the object to our PCA data and then re-run the kmeans using the predict. After fitting the kmeans object to population data then we use the predict function to come up with the cluster for the population data.

# In[45]:


get_ipython().run_cell_magic('time', '', '\nazdias_kmeans = KMeans(7)\nazdias_model = azdias_kmeans.fit(azdias_pca_200)# why did we fit here only\nazdias_labels = azdias_model.predict(azdias_pca_200)# why we ran the file again here in the predict? Osama')


# In[123]:


#a function to see the top weighst of each cluster
def explain_comp(cluster, n_weight):
    over_rep = pd.DataFrame.from_dict(dict(zip(azdias_clean.columns, 
    pca.inverse_transform(azdias_model.cluster_centers_[cluster]))), orient='index').rename(columns={0: 'feature_values'}).sort_values('feature_values', ascending=False)
    pd.concat((over_rep['feature_values'][:n_weight], over_rep['feature_values'][-n_weight:]), axis=0).plot(kind='barh')


# In[124]:


explain_comp(0, 10)


# In[129]:


explain_comp(5, 10)


# In[130]:


explain_comp(1, 10)


# In[131]:


explain_comp(6,10)


# Then we get the customer cleaned file and pass it to the pca object we created for the population file, and get back our customer PCA data.
# we then use the newly formed kmeans object based on the population data and pass it to the predict function of kmeans and get the labels of predicted clusters for the customer data

# In[58]:


get_ipython().run_cell_magic('time', '', 'customer_pca_200 = pca.transform(customers_clean) #why did we use only tranform here and not fit_transform? osama')


# In[91]:


customer_pca_200.shape


# In[60]:


customer_labels = azdias_kmeans.predict(customer_pca_200)


# In[ ]:


print(azdias_final.shape)
print(customers_final.shape)
print(azdias_labels.shape)
print(customer_labels.shape)
print(azdias_pca_200.shape)
print(customer_pca_200.shape)


# <b>We now create a DataFrame for the azdias_pca_200 clusters with labels </b>

# In[92]:


azdias_cluster_df = pd.DataFrame(azdias_pca_200, columns=comp)
azdias_cluster_df['cluster_labels'] = azdias_labels
azdias_cluster_df.tail()


# <b>We also create a DataFrame for the customer_200 clusters with labels </b>

# In[93]:


customer_cluster_df = pd.DataFrame(customer_pca_200, columns=comp)
customer_cluster_df['cluster_labels'] = customer_labels
customer_cluster_df.tail()


# In[63]:


customer_cluster_df.shape


# In[64]:


azdias_cluster_df.shape


# In[65]:


print(customer_cluster_df['cluster_labels'].value_counts())
print(azdias_cluster_df['cluster_labels'].value_counts())


# In[66]:


label_count_customer = customer_cluster_df['cluster_labels'].value_counts()
label_count_azdias = azdias_cluster_df['cluster_labels'].value_counts()


# In[67]:


plt.figure(figsize = (10,10))
plt.bar(label_count_customer.index, label_count_customer, label='customer_clusters')
plt.bar(label_count_azdias.index, label_count_azdias, width=0.5, label='general_pop_clusters')
plt.legend()
plt.title('Customer and Population cluster comparison')
plt.xlabel('Clusters')
plt.show()


# <b> In this step will choose the highest weights of the features and take the most over represented and the least represented and compare the means of the highest feature weights</b>

# Cluster 0 is the overrepresented, and i will reverse the PCA and scaler matrix back to its original numbers to better compare the means of the choosen features

# In[95]:


reverse_cluster_df_label_0 = customer_cluster_df[customer_cluster_df['cluster_labels'] == 0]
reverse_cluster_df_label_0.head()


# In[96]:


reverse_cluster_df_label_0_droped = reverse_cluster_df_label_0.drop('cluster_labels', axis=1)
cluster_0_pca = pca.inverse_transform(reverse_cluster_df_label_0_droped)
cluster_0_scaler = scaler.inverse_transform(cluster_0_pca)


# In[97]:


cluster_0_final = pd.DataFrame(cluster_0_scaler, columns=customers_clean.columns)
cluster_0_final.shape


# In[135]:


cluster_0_final.shape


# In[137]:


azdias_model.cluster_centers_[0].shape


# In[138]:


azdias_kmeans


# In[98]:


cluster_0_final[imp_cols].describe()


# 
# Now with the least represented cluster which is 2, we will do the same, reverse the PCA and scale back to the original choosen features and compare their means

# In[99]:


reverse_cluster_df_label_1 = customer_cluster_df[customer_cluster_df['cluster_labels'] == 1]
reverse_cluster_df_label_1.head()


# In[100]:


reverse_cluster_df_label_1_droped = reverse_cluster_df_label_1.drop('cluster_labels', axis=1)
cluster_1_pca = pca.inverse_transform(reverse_cluster_df_label_1_droped)
cluster_1_scaler = scaler.inverse_transform(cluster_1_pca)


# In[101]:


cluster_1_scaler.shape


# In[102]:


cluster_1_final = pd.DataFrame(cluster_1_scaler, columns=customers_clean.columns)
cluster_1_final.head()


# In[104]:


cluster_1_final[imp_cols].describe()


# In[107]:


over_rep = cluster_0_final[imp_cols].describe().loc['mean']
over_rep

under_rep = cluster_1_final[imp_cols].describe().loc['mean']
under_rep


# In[108]:


clusters_2_and_mean = pd.concat([over_rep, under_rep], axis=1)


# In[109]:


clusters_2_and_mean.columns = ['Target', 'Not_Target']


# In[114]:


clusters_2_and_mean.head()


# <b> Compare the means and analyse which features shows a high difference between the target group and the non-target group<b/>

# In[116]:


fig, ax = plt.subplots(figsize=(20,12))
plt.title("Mean Values of Columns Between Target and Non-Target Customers", fontsize=16)
plt.xlabel("Mean", fontsize=12)
clusters_2_and_mean.head().plot.bar(ax=ax)
plt.xticks(rotation=45)


# <b>Conclusion</b>
# We can see from the figure above that the target customers are high money savers: FINANZ_SPARER
# 
# financial typology: money saver	-1	unknown
# 	1	very high
# 	2	high
# 	3	average
# 	4	low
# 	5	very low
# 
# 
# And the non-target is not and for the mobility features, our target should be people who are low in mobility: MOBI_REGIO
# moving patterns	1	very high mobility
# 	2	high mobility
# 	3	middle mobility
# 	4	low mobility
# 	5	very low mobility 
# 	6	none
# 
# 

# <b> Heatmap to better explain</b>
# 
# We will be using a Heatmap to better explain the correlation between the clusters and the components, and also prepare the data from the kmeans model. 
# 

# In[35]:


azdias_model.cluster_centers_.shape


# In[ ]:


plt.figure(figsize = (20,12))
ax = sns.heatmap(azdias_model.cluster_centers_.T[0:10], cmap = 'YlGnBu', annot=True)
ax.set_xlabel("Cluster")
ax.set_ylabel("Components")
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
ax.set_title("Attribute Value by Centroid")
plt.show()


# In[164]:


t = pca.inverse_transform(azdias_model.cluster_centers_)


# In[171]:


t2 = pd.DataFrame(t, columns=azdias_clean.columns)
t2


# In[173]:


plt.figure(figsize = (25,30))
sns.heatmap(t2.T, cmap = 'YlGnBu')
xlabel('Features')
ylabel("clusters")
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 10)
plt.show()


# <b>The above heat map will help in interpreting some of the clusters:</b>
# 
# Based on our bar plots: The similar clusters are 4, 5, 1 and the dissimilar are 2,6,3
# 
# 
# 
#     
#     
#     
#     

# In[101]:


display_comp2(components_200_df, 2, 10)


# 
# 
# 

# In[144]:


display_comp2(components_200_df, 1, 10)


# 

# In[133]:


display_comp2(components_200_df, 0, 10)


# In[ ]:





# ## Part 2: Supervised Learning Model
# 
# Now that you've found which parts of the population are more likely to be customers of the mail-order company, it's time to build a prediction model. Each of the rows in the "MAILOUT" data files represents an individual that was targeted for a mailout campaign. Ideally, we should be able to use the demographic information from each individual to decide whether or not it will be worth it to include that person in the campaign.
# 
# The "MAILOUT" data has been split into two approximately equal parts, each with almost 43 000 data rows. In this part, you can verify your model with the "TRAIN" partition, which includes a column, "RESPONSE", that states whether or not a person became a customer of the company following the campaign. In the next part, you'll need to create predictions on the "TEST" partition, where the "RESPONSE" column has been withheld.

# In[2]:


mailout_train = pd.read_csv('../../data/Term2/capstone/arvato_data/Udacity_MAILOUT_052018_TRAIN.csv', sep=';')


# In[3]:


mailout_train.shape


# In[4]:


mailout_train.columns


# <b> Imbalanced Response Vector</b>
#     
# When reviewing the RESPONSE variable we can see clearly that its imbalanced, the Positive is 1.2% only of the total. Based on such a response percentage a accuracy metric must be chosen accordingly. The best suited is the ROC. The ROC will show clearly the TP and TN ratios. If we used another metric these details won’t show. Due to the imbalance, the model will be biased.
# 

# In[5]:


mailout_train['RESPONSE'].unique()


# In[6]:


#checking the Response ratio
mailout_train['RESPONSE'].value_counts()


# In[7]:


mailout_train['RESPONSE'].unique()


# In[8]:


mailout_train['RESPONSE'].value_counts()


# In[8]:


#to see if response variable is balanced or not

num_of_row = mailout_train.shape[0]
num_of_row


# In[9]:


#the response ratio is vert low and hence needs further metrics
(mailout_train[mailout_train['RESPONSE'] == 1].shape[0] / mailout_train.shape[0]) * 100


# <b> Preprocess the data for Machine Learning</b>
# 
# mailout_train data needs to go through the preprocessing steps before fitting it to any model, i will use the same function used at the beginning of the exercise to clean the data.
# Then remove any categorical columns, and separate the response variable. Then impute the missing elements and then finally to standardize the data.
# 
# After these steps we can proceed with testing out which classifier best fits the data.
# 

# In[11]:


# highly imbalanced data set
sns.countplot(mailout_train['RESPONSE'])


# In[15]:


mailout_train.shape


# In[12]:


#read in the file, same process used before
excel_att = pd.read_excel('DIAS Attributes - Values 2017.xlsx')
del excel_att['Unnamed: 0']
excel_att_n = excel_att['Attribute'].fillna(method='ffill')
excel_att['Attribute'] = excel_att_n
excel_att_table = excel_att[(excel_att['Meaning'].str.contains('unknown')) | (excel_att['Meaning'].str.contains('no'))]
#excel_att_table.head()

unknowns = []

for att in excel_att_table['Attribute'].unique():
    _ = excel_att_table.loc[excel_att_table['Attribute'] == att, 'Value'].astype(str).str.cat(sep=',')
    _ = _.split(',')
    unknowns.append(_)
    
#unknowns

excell_attibutes_com = pd.concat([ pd.Series(excel_att_table['Attribute'].unique()) , pd.Series(unknowns)], axis=1)
excell_attibutes_com.columns= ['attribute', 'missing_or_unknown']
excell_attibutes_com.head()


# In[13]:


def preprocess_df_for_ML(df, att_df):
    #function to clean the df and retreive the required labels to
    # convert the unkowns to correct NaN
    
    #first to convert all dtypes to float and keep object type same
    for col in df.columns:
        if df[col].dtype == np.int64:
            df[col] = df[col].astype(np.float64)
            
    #to retrieve the labels (unkown) and change to nan
    for row in att_df['attribute']:
        if row in df.columns:
            nan_list = att_df.loc[att_df['attribute'] == row, 'missing_or_unknown'].values[0]
            nan_index = df.loc[:, row].isin(nan_list)
            df.loc[nan_index, row] = np.nan
        else:
            continue
            
    columns_to_drop = ['AGER_TYP', 'ALTER_HH', 'ALTER_KIND1', 'ALTER_KIND2', 'ALTER_KIND3',
       'ALTER_KIND4', 'D19_BANKEN_ANZ_12', 'D19_BANKEN_ANZ_24',
       'D19_BANKEN_DATUM', 'D19_BANKEN_OFFLINE_DATUM',
       'D19_BANKEN_ONLINE_DATUM', 'D19_BANKEN_ONLINE_QUOTE_12',
       'D19_GESAMT_ANZ_12', 'D19_GESAMT_ANZ_24', 'D19_GESAMT_DATUM',
       'D19_GESAMT_OFFLINE_DATUM', 'D19_GESAMT_ONLINE_DATUM',
       'D19_GESAMT_ONLINE_QUOTE_12', 'D19_TELKO_ANZ_12', 'D19_TELKO_ANZ_24',
       'D19_TELKO_DATUM', 'D19_TELKO_OFFLINE_DATUM', 'D19_TELKO_ONLINE_DATUM',
       'D19_VERSAND_ANZ_12', 'D19_VERSAND_ANZ_24', 'D19_VERSAND_DATUM',
       'D19_VERSAND_OFFLINE_DATUM', 'D19_VERSAND_ONLINE_DATUM',
       'D19_VERSAND_ONLINE_QUOTE_12', 'D19_VERSI_ANZ_12', 'D19_VERSI_ANZ_24',
       'EXTSEL992', 'KBA05_ANHANG', 'KBA05_ANTG1', 'KBA05_ANTG2',
       'KBA05_ANTG3', 'KBA05_ANTG4', 'KBA05_BAUMAX', 'KBA05_CCM4', 'KBA05_KW3',
       'KBA05_MAXVORB', 'KBA05_MOD1', 'KBA05_MOD8', 'KBA05_MOTRAD',
       'KBA05_SEG1', 'KBA05_SEG5', 'KBA05_SEG6', 'KBA05_SEG7', 'KBA05_SEG8',
       'KBA05_SEG9', 'KK_KUNDENTYP', 'PLZ8_ANTG4', 'TITEL_KZ']
    
    #cutt_off_30 = df.columns[df.isnull().sum() / df.shape[0] > percentage]           
    
    #columns to drop based on theshold chosen
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    #row to be dropbed based on the number argument
    #row_to_drop = df.index[df.isnull().sum(axis=1) > 35]
    
    #df.drop(row_to_drop, axis=0, inplace=True)
    
    

    return df


# In[15]:


#we must preprocess the data first
mailout_train_pre = preprocess_df_for_ML(mailout_train, excell_attibutes_com)
mailout_train_pre.head()


# <b>Remove the Categorical cols</b>

# In[23]:


cat_to_remove = mailout_train_pre.columns[mailout_train_pre.dtypes == 'object']
cat_to_remove


# In[24]:


mailout_train_pre_nocat = mailout_train_pre.drop(cat_to_remove, axis=1)


# In[25]:


mailout_train_pre_nocat.drop('RESPONSE', axis=1, inplace=True)


# In[26]:


mailout_train_pre_nocat.shape


# <b> Create the Pipeline </b>
# 
# Start by creating the objects to impute and standardize the data, and then i create the pipeline object that will help in processing the files in better fashion.

# In[27]:


#ini imputer and scaler objects with default param
imputer  = Imputer()
scaler = StandardScaler()


# In[28]:


def create_piping_object():
    pipe = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler)
    ])
    
    return pipe


# In[29]:


piper = create_piping_object()


# Pipeline object ready to be used to transform any Data Frame

# In[30]:


piper


# In[31]:


mailout_ready_piped = piper.fit_transform(mailout_train_pre_nocat)


# Convert the numpy matrix to DF for better readability

# In[32]:


mailout_ready_piped_df = pd.DataFrame(mailout_ready_piped, columns=mailout_train_pre_nocat.columns)


# In[33]:


mailout_ready_piped_df.shape


# In[34]:


mailout_train_pre['RESPONSE'].head()


# <b> Model Performance </b>
# 
# Due to the response variable nature, a binary classifier will be used. I will create a function to run multiple models and see the best ROC metric, i will also use Cross Validation technique to help in generalization and not fitting to noise. The link below provides a clear explanation of CV.
# 
# The ROC metric was chose due to the imbalanced data set, we have only 1.2% positive (1) response. This is a very low number and we can’t use the accuracy metric to provide a clear score for the models.
# 
# https://medium.com/lumiata/cross-validation-for-imbalanced-datasets-9d203ba47e8

# <b> Trying multiple Algorithms</b>
# 
# By creating a function that can test the ROC accuracy of three different classifiers, then based on the score dict, we can choose the best model to then optimize with hyper parameters.
# 
# The three classifiers to be used are:
# 
# AdaBoostClassifier - 
# GradientBoostingClassifier - 
# LogisticRegressionCV 

# In[35]:


#function to test out the baseline and other classifiers
score_list = {}

def run_algo_2(X_df, y_df):
    
    clf_a = AdaBoostClassifier(random_state=28)
    clf_b = GradientBoostingClassifier(random_state=28)
    clf_c = LogisticRegressionCV(random_state=28)


    for clf in [clf_a, clf_b, clf_c]:
        mean_scores = clf
        print('working on the algo{}'.format(clf))
        algo_mean_scores = cross_val_score(clf, X_df, y_df, cv=5, scoring='roc_auc').mean()
        
        
    #print(score_list)
        score_list[mean_scores] = algo_mean_scores


# After testing the three classifers, GradientBoostingClassifier shows ths highest ROC score at 75%

# In[83]:


get_ipython().run_cell_magic('time', '', "run_algo_2(mailout_ready_piped_df, mailout_train['RESPONSE'])")


# In[104]:


score_list.values()


# <b> Hyper parameter evaluation</b>
# At this stage we try different hyper parameters to increase our score a little further if possible
# 

# In[ ]:


#below is the params to test for and see the best predictor
param_grid = {'learning_rate':[0.1, 0.2],
             'n_estimators': [100],
             'max_depth' : [3,5],
              'min_samples_split': [2, 4]}


# In[26]:


get_ipython().run_cell_magic('time', '', "grid = GridSearchCV(GradientBoostingClassifier(random_state=28), param_grid, scoring='roc_auc', cv=5, n_jobs=-1)")


# In[27]:


grid.fit(mailout_ready_piped_df, mailout_train['RESPONSE'])


# In[28]:


#to see all params that can be evaulated 
grid.best_estimator_


# In[30]:


grid.best_score_


# In[29]:


#these are the best param choosen by the model. These will be used later to ini the new model.
grid.best_params_


# <b> Model based on best parameters</b>
# 
# After trying the types of classifiers, the highest model result is the GradientBoostingClassifier at ROC score of 0.75, but it comes with some disadvantage, the speed of the fitting.
# 
# After choosing the model, i ran a grid search to come up with the best parameters. Hence to initiate the model with the new parameters.
# 

# In[36]:


#ini the model with best param
GBC_tuned = GradientBoostingClassifier(learning_rate=0.1, 
                                       max_depth=3, min_samples_split=4,
                                      n_estimators=100)


# In[37]:


get_ipython().run_cell_magic('time', '', "GBC_tuned.fit(mailout_ready_piped_df, mailout_train['RESPONSE'])")


# ## Part 3: Kaggle Competition
# 
# Now that you've created a model to predict which individuals are most likely to respond to a mailout campaign, it's time to test that model in competition through Kaggle. If you click on the link [here](http://www.kaggle.com/t/21e6d45d4c574c7fa2d868f0e8c83140), you'll be taken to the competition page where, if you have a Kaggle account, you can enter. If you're one of the top performers, you may have the chance to be contacted by a hiring manager from Arvato or Bertelsmann for an interview!
# 
# Your entry to the competition should be a CSV file with two columns. The first column should be a copy of "LNR", which acts as an ID number for each individual in the "TEST" partition. The second column, "RESPONSE", should be some measure of how likely each individual became a customer – this might not be a straightforward probability. As you should have found in Part 2, there is a large output class imbalance, where most individuals did not respond to the mailout. Thus, predicting individual classes and using accuracy does not seem to be an appropriate performance evaluation method. Instead, the competition will be using AUC to evaluate performance. The exact values of the "RESPONSE" column do not matter as much: only that the higher values try to capture as many of the actual customers as possible, early in the ROC curve sweep.

# In[38]:


mailout_test = pd.read_csv('../../data/Term2/capstone/arvato_data/Udacity_MAILOUT_052018_TEST.csv', sep=';')


# <b> Preprocess the testing data</b>
# 
# Now we have to clean and preprocess the mailout_test file in a similar fashion to the mailout_train. The difference here is there is no response variable.
# 
# After cleaning the mailout_test file, then we can use our model and predict the response variable.
# 

# In[39]:


mailout_test.shape


# In[40]:


#checking if the df has nans
mailout_test.isnull().sum().head()


# In[42]:


mailout_test.head()


# In[43]:


#re reun the function to clean the file
mailout_test_pre = preprocess_df_for_ML(mailout_test, excell_attibutes_com)


# In[44]:


mailout_test_pre.shape


# In[45]:


#Remove the catigorical features
cat_to_remove = mailout_test_pre.columns[mailout_test_pre.dtypes == 'object']
cat_to_remove


# In[46]:


mailout_test_pre_no_cat = mailout_test_pre.drop(cat_to_remove, axis=1)


# In[47]:


#checking the shape of the df, to make sure its like the mailout_train
mailout_test_pre_no_cat.shape


# In[157]:


print(mailout_test_pre.columns)


# In[48]:


#reuse the pipeline to impute and scale
mailout_test_piped = piper.fit_transform(mailout_test_pre_no_cat)


# In[49]:


mailout_test_piped.shape


# In[56]:


#output is in a matrix numpy format
mailout_test_piped


# <b> Transformation </b>
# Transform the imputed and scaled matrix to DF format

# In[50]:


mailout_test_piped_df = pd.DataFrame(mailout_test_piped, columns=mailout_test_pre_no_cat.columns)


# Use the model to predict the responce variable based on our mailout_test file

# In[51]:


#using predict_proba to get a probabilty as a response
pred = GBC_tuned.predict_proba(mailout_test_piped_df)


# In[104]:


pred_Df = pd.DataFrame(index=mailout_test['LNR'], data=pred)


# In[105]:


pred_Df.rename(index=str, columns={0:'First', 1:'Response'}, inplace=True)
# so go for 0 or 1 as a response


# In[111]:


pred_Df.head()


# In[112]:


pred_Df.drop('First', axis=1, inplace=True)


# In[113]:


#save file as csv for submission
for_submission = pred_Df.to_csv('kaggle_submit.csv', index=False)


# <b> Feature Importance</b>
# 
# At this stage we can use the methods available in the model to see the most important features by weight 
# 

# In[57]:


#check the len of features is the same as cols
len(GBC_tuned.feature_importances_)


# In[94]:


#function to display the most important features of the classifier

def feature_display(clf):
    GBC_tuned_features = pd.DataFrame(clf.feature_importances_)
    idx = np.argsort(GBC_tuned.feature_importances_)[::-1]
    Features_df = pd.concat([GBC_tuned_features.iloc[idx], pd.Series(mailout_ready_piped_df.columns[idx])], axis=1)
    Features_df.columns = ['weights', 'feature_name']
    Features_df_10 = Features_df.iloc[idx][:10]
    Features_df_10.plot.barh(x = 'feature_name',
                    y = 'weights')


# In[95]:


feature_display(GBC_tuned) # evaulate the best split - read about these features when highest are what they carry as attrubites


# <b> Summary</b>
# 
# The above plot shows that SEMIO_RAT is the largest weight the separates the data at best. Second comes the next in line, RETOURTYP_BK_S. These features with the highest weighs best explain how the tree is splitting the customer as a high response or low response. 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[159]:





# In[81]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[184]:


def feature_plot(importances, X_train, y_train, num_feat=5):
     
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:num_feat]]
    values = importances[indices][:num_feat]
 
    # Create the plot
    fig = plt.figure(figsize = (16,9))
    plt.title("Normalized Weights for the Most Predictive Features", fontsize = 16)
    plt.barh(np.arange(num_feat), values[::-1], height = 0.6, align="center",           label = "Feature Weight")
    plt.barh(np.arange(num_feat) - 0.3, np.cumsum(values)[::-1], height = 0.2, align = "center",           label = "Cumulative Feature Weight")
    plt.yticks(np.arange(num_feat), columns[::-1])
    plt.xlabel("Weight", fontsize = 12)
    plt.ylabel('')
     
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.show()


# In[ ]:





# In[47]:


GradientBoostingClassifier.feature_importances_


# In[ ]:


# finish the project report, export image to explain better on the report

