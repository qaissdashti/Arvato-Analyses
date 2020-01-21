
# Capstone Project: Create a Customer Segmentation Report for Arvato Financial Services

In this project, you will analyze demographics data for customers of a mail-order sales company in Germany, comparing it against demographics information for the general population. You'll use unsupervised learning techniques to perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company. Then, you'll apply what you've learned on a third dataset with demographics information for targets of a marketing campaign for the company, and use a model to predict which individuals are most likely to convert into becoming customers for the company. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.

If you completed the first term of this program, you will be familiar with the first part of this project, from the unsupervised learning project. The versions of those two datasets used in this project will include many more features and has not been pre-cleaned. You are also free to choose whatever approach you'd like to analyzing the data rather than follow pre-determined steps. In your work on this project, make sure that you carefully document your steps and decisions, since your main deliverable for this project will be a blog post reporting your findings.


```python
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
%matplotlib inline
```

    /opt/conda/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    

## Part 0: Get to Know the Data

There are four data files associated with this project:

- `Udacity_AZDIAS_052018.csv`: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
- `Udacity_CUSTOMERS_052018.csv`: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
- `Udacity_MAILOUT_052018_TRAIN.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
- `Udacity_MAILOUT_052018_TEST.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).

Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. Use the information from the first two files to figure out how customers ("CUSTOMERS") are similar to or differ from the general population at large ("AZDIAS"), then use your analysis to make predictions on the other two files ("MAILOUT"), predicting which recipients are most likely to become a customer for the mail-order company.

The "CUSTOMERS" file contains three extra columns ('CUSTOMER_GROUP', 'ONLINE_PURCHASE', and 'PRODUCT_GROUP'), which provide broad information about the customers depicted in the file. The original "MAILOUT" file included one additional column, "RESPONSE", which indicated whether or not each recipient became a customer of the company. For the "TRAIN" subset, this column has been retained, but in the "TEST" subset it has been removed; it is against that withheld column that your final predictions will be assessed in the Kaggle competition.

Otherwise, all of the remaining columns are the same between the three data files. For more information about the columns depicted in the files, you can refer to two Excel spreadsheets provided in the workspace. [One of them](./DIAS Information Levels - Attributes 2017.xlsx) is a top-level list of attributes and descriptions, organized by informational category. [The other](./DIAS Attributes - Values 2017.xlsx) is a detailed mapping of data values for each feature in alphabetical order.

In the below cell, we've provided some initial code to load in the first two datasets. Note for all of the `.csv` data files in this project that they're semicolon (`;`) delimited, so an additional argument in the [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call has been included to read in the data properly. Also, considering the size of the datasets, it may take some time for them to load completely.

You'll notice when the data is loaded in that a warning message will immediately pop up. Before you really start digging into the modeling and analysis, you're going to need to perform some cleaning. Take some time to browse the structure of the data and look over the informational spreadsheets to understand the data values. Make some decisions on which features to keep, which features to drop, and if any revisions need to be made on data formats. It'll be a good idea to create a function with pre-processing steps, since you'll need to clean all of the datasets before you work with them.


```python
# load in the data

azdias = pd.read_csv('../../data/Term2/capstone/arvato_data/Udacity_AZDIAS_052018.csv', sep=';')
customers = pd.read_csv('../../data/Term2/capstone/arvato_data/Udacity_CUSTOMERS_052018.csv', sep=';')
```

    /opt/conda/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2785: DtypeWarning: Columns (18,19) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    


```python
azdias.shape
```




    (891221, 366)




```python
customers.shape
```




    (191652, 369)




```python
sns.countplot(azdias.isnull().sum())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f8c4b95fb00>




![png](output_6_1.png)



```python
sns.countplot(customers.isnull().sum())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f8c4ec57588>




![png](output_7_1.png)


## Part 1: Customer Segmentation Report

The main bulk of your analysis will come in this part of the project. Here, you should use unsupervised learning techniques to describe the relationship between the demographics of the company's existing customers and the general population of Germany. By the end of this part, you should be able to describe parts of the general population that are more likely to be part of the mail-order company's main customer base, and which parts of the general population are less so.

<b> Steps to be taken at preprocessing stage:</b>

Excel file cleanup:
1. Read in the excel file
2. Fill forward the empty lines with the line above for better element retrieval
3. Retrieve the missing and or the unknown labels and place in list
4. Split the missing and unknown labels and loop through them to get the unique elements only
5. Concat the attribute names with a list of the unique elements


Within the processing function the below will take place for both files of azdias and customer:
1. All ints must be converted to floats
2. Check is row in the file with the same attribute in the excel file, if available the retrieve the element if not then NAN
3. Use separate function to choose the cutoff level, then place the cols to be dropped in the preprocessing function. columns already placed in function
4. choose the number of row needed to be removed as a custoff, also can be calculated in choose_cut_off_row function.
5. All changes are in place and a df is returned

Outside the preprocessing function step:
1. to remove the categorical cols
2. to test out using both functions "choose_cut_off" and "choose_cut_off_row" to choose the cols and rows that need be removed
3. the three cols PRODUCT_GROUP CUSTOMER_GROUP ONLINE_PURCHASE, must be removed separately

Note: to ensure that both files have the similar cols names and same number of features before moving further.





```python
customers.shape
```




    (191652, 369)




```python
azdias.shape
```




    (891221, 366)




```python
# convert all dtypes of int to float, and keep the object type - testing part
for col in azdias.columns:
    if azdias[col].dtype == np.int64:
        azdias[col] = azdias[col].astype(np.float64)
    
#check if all azdias dtypes are floats and objects
azdias.dtypes.value_counts()
```




    (191652, 369)



<b> We start with the excel cleanup</b>

<b>Convert and correct the excel file to ensure it can be used to clean and pull the needed attributes from for the azdias file</b>
    
first step to clean the excel sheet with the attributes to be used
for cleaning the general population file


```python
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
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>attribute</th>
      <th>missing_or_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGER_TYP</td>
      <td>[-1, 0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ALTERSKATEGORIE_GROB</td>
      <td>[-1,  0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ALTER_HH</td>
      <td>[0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ANREDE_KZ</td>
      <td>[-1,  0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BALLRAUM</td>
      <td>[-1]</td>
    </tr>
  </tbody>
</table>
</div>



<b>Loop to read in the labels or add NaNs</b>

Below is the loop to clean the azdias data by using the excel sheet provided and add a np.nan inplace of missing values that are being read from the excell_attibutes_com file



```python
#testing part
for row in excell_attibutes_com['attribute']:
    print(row)
    if row in azdias.columns:
        na_map = excell_attibutes_com.loc[excell_attibutes_com['attribute'] == row, 'missing_or_unknown'].values[0]
        na_idx = azdias.loc[:, row].isin(na_map)
        azdias.loc[na_idx, row] = np.nan
    else:
        continue

```

    AGER_TYP
    


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-114-15b28c310309> in <module>()
          2 for row in excell_attibutes_com['attribute']:
          3     print(row)
    ----> 4     if row in azdias.columns:
          5         na_map = excell_attibutes_com.loc[excell_attibutes_com['attribute'] == row, 'missing_or_unknown'].values[0]
          6         na_idx = azdias.loc[:, row].isin(na_map)
    

    NameError: name 'azdias' is not defined



```python
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
```


```python
%%time
azdias_pre = preprocess_df(azdias, excell_attibutes_com)
```

    CPU times: user 27.6 s, sys: 52.5 s, total: 1min 20s
    Wall time: 7min 34s
    


```python
%%time
customers_pre = preprocess_df(customers, excell_attibutes_com)
```

    CPU times: user 6.53 s, sys: 3.43 s, total: 9.96 s
    Wall time: 1min 7s
    


```python
azdias_pre.shape
```




    (736966, 313)




```python
customers_pre.shape
```




    (134197, 316)



Seperate the 3 additional cols of customer file


```python
three_col_df_customer = customers_pre[['PRODUCT_GROUP', 'CUSTOMER_GROUP', 'ONLINE_PURCHASE']]
three_col_df_customer.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PRODUCT_GROUP</th>
      <th>CUSTOMER_GROUP</th>
      <th>ONLINE_PURCHASE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COSMETIC_AND_FOOD</td>
      <td>MULTI_BUYER</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>COSMETIC_AND_FOOD</td>
      <td>MULTI_BUYER</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>COSMETIC</td>
      <td>MULTI_BUYER</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FOOD</td>
      <td>MULTI_BUYER</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>COSMETIC_AND_FOOD</td>
      <td>MULTI_BUYER</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
customers_pre.shape
```




    (134197, 313)




```python
#for customer file only

def clean_customer_after_pre(cust_df):
    del cust_df['PRODUCT_GROUP']
    del cust_df['CUSTOMER_GROUP']
    del cust_df['ONLINE_PURCHASE']
    
    customers_pre_no_cat = remove_cat(cust_df)
    return customers_pre_no_cat    
```

<b>Categorical columns</b> will be removed due to their labels, and that most have high number of different labels


```python
#categoricals columns  must be removed manualy!! for both Azdias and Customer file

def remove_cat(df):
    df.columns[df.dtypes == 'object']
    catigorical_col = df.columns[df.dtypes == 'object']
    df.drop(catigorical_col, axis=1, inplace=True)
    return df
```


```python
customers_pre_no_cat = clean_customer_after_pre(customers_pre)
```


```python
customers_pre_no_cat.shape
```




    (134197, 307)




```python
azdias_pre_no_cat = remove_cat(azdias_pre)

```


```python
azdias_pre_no_cat.shape
```




    (736966, 307)




```python
# to better analyse and decide on the cat columns and see if to keep or remove entirly. 
for col in azdias.columns[azdias.dtypes == 'object']:
    print(col)
    print(azdias[col].unique())
    
```

<b>Removing NaNs based on a cuttoff ratio</b>

The below two function are used to choose the percentage and number of NaNs elements to be removed - also for testing before using the preprocessing function



```python
def choose_cut_off(df, percentage):
    #returns the cols that will be removed based on the % number placed in percentage
    cutt_off_30 = df.columns[df.isnull().sum() / df.shape[0] > percentage]           
    
    #columns to drop based on theshold chosen
    df.drop(cutt_off_30, axis=1, inplace=True)
```


```python
def choose_cut_off_row(df, number):
    row_to_drop = df.index[df.isnull().sum(axis=1) > number]
    
    df.drop(row_to_drop, axis=0, inplace=True)
    
```

<b>Pickling the files for better space managemnet</b>

Import and use pickle, to save and serlize the files for better space resources. After cleaning the files we will pickle them so they can be downloaded fast and to be used right away.


```python
#to upload cleaned files with pickle

#pickle.dump(azdias, open("azdias_clean.pickle", "wb"))

# Dump the customers dataframe to a pickle object to use for later.
#pickle.dump(customers, open("customers_clean.pickle", "wb"))
```


```python
# to reload uploaded files back 
#azdias_clean = pickle.load(open("azdias_clean.pickle", "rb"))
#customers_clean = pickle.load(open("customers_clean.pickle", "rb"))
```


```python
azdias_clean.shape
```




    (798219, 307)




```python
customers_clean.shape
```




    (141743, 307)



<b> Assert test</b>
Test to see if both files have the same number of features before PCA


```python
def test_df(azdias_df, cust_df):
    assert np.all(cust_df.columns) == np.all(azdias_df.columns)
    assert sum(cust_df.columns == azdias_df.columns) ==307
    print('both files are ready for PCA, with {} columns'.format(len(customers_pre_no_cat.columns)))
```


```python
test_df(azdias_pre_no_cat, customers_pre_no_cat)
```

    both files are ready for PCA, with 307 columns
    

<b>Imputing the Nans</b>
After cleaning and removing the large number of NaNs in the cols and row, we now impute the remaining nans with the most frequent element in the column.


```python
def impute(az_df, cust_df):
    
# imputer_test = Imputer(strategy='mean or most_frequent', axis=0) to impute col wise
#test_df_1 = pd.DataFrame(imputer_test.fit_transform(test_df_1))
    imputer = Imputer(strategy='most_frequent', axis=0)

    azdias_clean = pd.DataFrame(imputer.fit_transform(az_df),
                                columns=az_df.columns)

    customers_clean = pd.DataFrame(imputer.fit_transform(cust_df),
                                   columns=cust_df.columns)
    
    return azdias_clean, customers_clean

```


```python
azdias_clean_pca_ready, customer_clean_pca_ready = impute(azdias_pre_no_cat, customers_pre_no_cat)

```


```python
azdias_clean_pca_ready.shape
```




    (736966, 307)




```python
customer_clean_pca_ready.shape
```




    (134197, 307)




```python
customer_clean_pca_ready.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LNR</th>
      <th>AKT_DAT_KL</th>
      <th>ALTERSKATEGORIE_FEIN</th>
      <th>ANZ_HAUSHALTE_AKTIV</th>
      <th>ANZ_HH_TITEL</th>
      <th>ANZ_KINDER</th>
      <th>ANZ_PERSONEN</th>
      <th>ANZ_STATISTISCHE_HAUSHALTE</th>
      <th>ANZ_TITEL</th>
      <th>ARBEIT</th>
      <th>...</th>
      <th>VHN</th>
      <th>VK_DHT4A</th>
      <th>VK_DISTANZ</th>
      <th>VK_ZG11</th>
      <th>W_KEIT_KIND_HH</th>
      <th>WOHNDAUER_2008</th>
      <th>WOHNLAGE</th>
      <th>ZABEOTYP</th>
      <th>ANREDE_KZ</th>
      <th>ALTERSKATEGORIE_GROB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9626.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>143872.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>143873.0</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>143874.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>143888.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 307 columns</p>
</div>




```python
azdias_clean_pca_ready.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LNR</th>
      <th>AKT_DAT_KL</th>
      <th>ALTERSKATEGORIE_FEIN</th>
      <th>ANZ_HAUSHALTE_AKTIV</th>
      <th>ANZ_HH_TITEL</th>
      <th>ANZ_KINDER</th>
      <th>ANZ_PERSONEN</th>
      <th>ANZ_STATISTISCHE_HAUSHALTE</th>
      <th>ANZ_TITEL</th>
      <th>ARBEIT</th>
      <th>...</th>
      <th>VHN</th>
      <th>VK_DHT4A</th>
      <th>VK_DISTANZ</th>
      <th>VK_ZG11</th>
      <th>W_KEIT_KIND_HH</th>
      <th>WOHNDAUER_2008</th>
      <th>WOHNLAGE</th>
      <th>ZABEOTYP</th>
      <th>ANREDE_KZ</th>
      <th>ALTERSKATEGORIE_GROB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>910220.0</td>
      <td>9.0</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>910225.0</td>
      <td>9.0</td>
      <td>17.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>910226.0</td>
      <td>1.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>910241.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>910244.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 307 columns</p>
</div>



<b>Standardizing</b>

Before going ahead with the PCA, we need to ensure that both files are standardized and we will use the StandardScaler object here



```python
scaler = StandardScaler()
azdias_clean = pd.DataFrame(scaler.fit_transform(azdias_clean_pca_ready), columns = azdias_clean_pca_ready.columns)
customers_clean = pd.DataFrame(scaler.transform(customer_clean_pca_ready), columns = customer_clean_pca_ready.columns)
```


```python
azdias_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LNR</th>
      <th>AKT_DAT_KL</th>
      <th>ALTERSKATEGORIE_FEIN</th>
      <th>ANZ_HAUSHALTE_AKTIV</th>
      <th>ANZ_HH_TITEL</th>
      <th>ANZ_KINDER</th>
      <th>ANZ_PERSONEN</th>
      <th>ANZ_STATISTISCHE_HAUSHALTE</th>
      <th>ANZ_TITEL</th>
      <th>ARBEIT</th>
      <th>...</th>
      <th>VHN</th>
      <th>VK_DHT4A</th>
      <th>VK_DISTANZ</th>
      <th>VK_ZG11</th>
      <th>W_KEIT_KIND_HH</th>
      <th>WOHNDAUER_2008</th>
      <th>WOHNLAGE</th>
      <th>ZABEOTYP</th>
      <th>ANREDE_KZ</th>
      <th>ALTERSKATEGORIE_GROB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.057651</td>
      <td>1.262163</td>
      <td>1.560803</td>
      <td>0.187007</td>
      <td>-0.125301</td>
      <td>-0.299103</td>
      <td>0.224541</td>
      <td>0.335620</td>
      <td>-0.060663</td>
      <td>-0.183227</td>
      <td>...</td>
      <td>1.375897</td>
      <td>0.695673</td>
      <td>1.065141</td>
      <td>1.455824</td>
      <td>-0.747533</td>
      <td>0.547694</td>
      <td>-0.028626</td>
      <td>1.141418</td>
      <td>0.958542</td>
      <td>-1.682222</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.057670</td>
      <td>1.262163</td>
      <td>0.674370</td>
      <td>0.120808</td>
      <td>-0.125301</td>
      <td>-0.299103</td>
      <td>-0.634577</td>
      <td>-0.031724</td>
      <td>-0.060663</td>
      <td>-0.183227</td>
      <td>...</td>
      <td>-0.355004</td>
      <td>1.046117</td>
      <td>0.451202</td>
      <td>0.024600</td>
      <td>-0.747533</td>
      <td>0.547694</td>
      <td>-1.083166</td>
      <td>1.141418</td>
      <td>0.958542</td>
      <td>0.168043</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.057674</td>
      <td>-0.934344</td>
      <td>-0.212063</td>
      <td>-0.474984</td>
      <td>-0.125301</td>
      <td>-0.299103</td>
      <td>-1.493696</td>
      <td>-0.399068</td>
      <td>-0.060663</td>
      <td>-1.187332</td>
      <td>...</td>
      <td>-2.085904</td>
      <td>0.345229</td>
      <td>0.758172</td>
      <td>1.813630</td>
      <td>0.952202</td>
      <td>0.547694</td>
      <td>1.553183</td>
      <td>-0.279476</td>
      <td>0.958542</td>
      <td>1.093175</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.057732</td>
      <td>-0.934344</td>
      <td>0.009546</td>
      <td>-0.342586</td>
      <td>-0.125301</td>
      <td>-0.299103</td>
      <td>1.942778</td>
      <td>-0.325599</td>
      <td>-0.060663</td>
      <td>0.820878</td>
      <td>...</td>
      <td>-0.355004</td>
      <td>-1.056547</td>
      <td>-0.776677</td>
      <td>-0.691012</td>
      <td>-1.314111</td>
      <td>0.547694</td>
      <td>-0.555896</td>
      <td>0.430971</td>
      <td>-1.043251</td>
      <td>0.168043</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.057744</td>
      <td>-0.934344</td>
      <td>-0.876887</td>
      <td>-0.210188</td>
      <td>-0.125301</td>
      <td>-0.299103</td>
      <td>-0.634577</td>
      <td>-0.399068</td>
      <td>-0.060663</td>
      <td>-1.187332</td>
      <td>...</td>
      <td>-0.355004</td>
      <td>1.396561</td>
      <td>-0.162737</td>
      <td>-0.691012</td>
      <td>0.952202</td>
      <td>0.547694</td>
      <td>1.553183</td>
      <td>0.430971</td>
      <td>0.958542</td>
      <td>-1.682222</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 307 columns</p>
</div>




```python
customers_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LNR</th>
      <th>AKT_DAT_KL</th>
      <th>ALTERSKATEGORIE_FEIN</th>
      <th>ANZ_HAUSHALTE_AKTIV</th>
      <th>ANZ_HH_TITEL</th>
      <th>ANZ_KINDER</th>
      <th>ANZ_PERSONEN</th>
      <th>ANZ_STATISTISCHE_HAUSHALTE</th>
      <th>ANZ_TITEL</th>
      <th>ARBEIT</th>
      <th>...</th>
      <th>VHN</th>
      <th>VK_DHT4A</th>
      <th>VK_DISTANZ</th>
      <th>VK_ZG11</th>
      <th>W_KEIT_KIND_HH</th>
      <th>WOHNDAUER_2008</th>
      <th>WOHNLAGE</th>
      <th>ZABEOTYP</th>
      <th>ANREDE_KZ</th>
      <th>ALTERSKATEGORIE_GROB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.436805</td>
      <td>-0.934344</td>
      <td>-0.876887</td>
      <td>-0.474984</td>
      <td>-0.125301</td>
      <td>-0.299103</td>
      <td>0.224541</td>
      <td>-0.472536</td>
      <td>-0.060663</td>
      <td>-2.191437</td>
      <td>...</td>
      <td>0.510447</td>
      <td>-0.355659</td>
      <td>-1.390616</td>
      <td>-1.406623</td>
      <td>0.952202</td>
      <td>0.547694</td>
      <td>1.553183</td>
      <td>-0.279476</td>
      <td>-1.043251</td>
      <td>1.093175</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.915908</td>
      <td>-0.934344</td>
      <td>-3.092970</td>
      <td>-0.474984</td>
      <td>-0.125301</td>
      <td>-0.299103</td>
      <td>-0.634577</td>
      <td>-0.472536</td>
      <td>-0.060663</td>
      <td>-0.183227</td>
      <td>...</td>
      <td>1.375897</td>
      <td>1.396561</td>
      <td>1.679081</td>
      <td>1.813630</td>
      <td>0.952202</td>
      <td>0.547694</td>
      <td>-1.083166</td>
      <td>-0.279476</td>
      <td>0.958542</td>
      <td>1.093175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.915904</td>
      <td>-0.934344</td>
      <td>-1.320104</td>
      <td>-0.541184</td>
      <td>-0.125301</td>
      <td>-0.299103</td>
      <td>-1.493696</td>
      <td>-0.472536</td>
      <td>-0.060663</td>
      <td>-2.191437</td>
      <td>...</td>
      <td>-0.355004</td>
      <td>-0.005215</td>
      <td>-1.083646</td>
      <td>-1.406623</td>
      <td>0.952202</td>
      <td>0.547694</td>
      <td>1.553183</td>
      <td>-1.700369</td>
      <td>-1.043251</td>
      <td>1.093175</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.915900</td>
      <td>-0.934344</td>
      <td>0.009546</td>
      <td>-0.077789</td>
      <td>-0.125301</td>
      <td>-0.299103</td>
      <td>1.942778</td>
      <td>-0.031724</td>
      <td>-0.060663</td>
      <td>-0.183227</td>
      <td>...</td>
      <td>1.375897</td>
      <td>-1.056547</td>
      <td>-0.776677</td>
      <td>-0.691012</td>
      <td>-1.314111</td>
      <td>0.547694</td>
      <td>-0.555896</td>
      <td>-1.700369</td>
      <td>-1.043251</td>
      <td>0.168043</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.915846</td>
      <td>-0.934344</td>
      <td>-0.876887</td>
      <td>-0.474984</td>
      <td>-0.125301</td>
      <td>-0.299103</td>
      <td>0.224541</td>
      <td>-0.472536</td>
      <td>-0.060663</td>
      <td>-0.183227</td>
      <td>...</td>
      <td>1.375897</td>
      <td>-1.757436</td>
      <td>-1.697586</td>
      <td>-1.764429</td>
      <td>0.952202</td>
      <td>0.547694</td>
      <td>-1.610436</td>
      <td>-0.989923</td>
      <td>-1.043251</td>
      <td>0.168043</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 307 columns</p>
</div>



<b>Files are now ready for PCA and KMEANS:</b>

Lets Pickle the files so, when we need to retive them we can do so without running all the past functions that take time.


```python
# Dump the azdias dataframe to a pickle object since it takes up so much room in memory.
pickle.dump(azdias_clean, open("azdias_final.pickle", "wb"))

# Dump the customers dataframe to a pickle object to use for later.
pickle.dump(customers_clean, open("customers_final.pickle", "wb"))
```


```python
# Reload cleaned azdias object as saved after above analysis (may need to rerun imports)
azdias_clean = pickle.load(open("azdias_final.pickle", "rb"))

# Reload cleaned customers object as saved after above analysis
customers_clean = pickle.load(open("customers_final.pickle", "rb"))
```


```python
azdias_clean.shape
```




    (736966, 307)




```python
customers_clean.shape
```




    (134197, 307)



<b>Dimensionality Reduction on the Data - PCA</b>

PCA here will help in reducing the feature space from 307 to 200. After plotting the new features of 200, we can calculate the sum of explained variance which comes to a total of 94% which is high and cover most of the data, and at least reduced it by 93 features


```python
pca = PCA(200)
```


```python
%%time

azdias_pca_200 = pca.fit_transform(azdias_clean)
```

    CPU times: user 2min 52s, sys: 1min 21s, total: 4min 14s
    Wall time: 18min 38s
    


```python
azdias_pca_200.shape
```




    (736966, 200)




```python
sum(pca.explained_variance_ratio_)
```




    0.94883491456121749




```python
len(pca.components_)
```




    200




```python
pca.components_.shape
```




    (200, 307)



The function below will take in the PCA object and plot the new number of components and how much explained variance they have. Our 200 newly formed features now have a explained variance of about 90%.

I have chosen 200 features after running the PCA with all the features and the plot showed that at about 200 the explained variance is above 85%.

Based on that i have decided to go with 200 components.



```python
def com_plot(pca):
    
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Explained Variance Per Principal Component')

```


```python
com_plot(pca)
```


![png](output_69_0.png)



```python
pca.components_
```




    array([[-0.02519649,  0.06899909,  0.0369175 , ...,  0.091885  ,
             0.00682688, -0.04770039],
           [ 0.0824931 ,  0.01286633, -0.00384746, ..., -0.00961505,
             0.00498181, -0.00206282],
           [ 0.02610313,  0.04350405,  0.16946174, ...,  0.04499787,
             0.01033779, -0.19010463],
           ..., 
           [ 0.01969613,  0.01718151, -0.05105555, ...,  0.04096604,
             0.03455326, -0.05729504],
           [ 0.0084195 , -0.0163854 ,  0.06416132, ..., -0.00242773,
             0.01245801, -0.04760116],
           [-0.0590122 ,  0.03539547, -0.06306534, ..., -0.03164444,
             0.03238647, -0.0203024 ]])




```python
components_200 = pca.components_
```


```python
components_200_df = pd.DataFrame(components_200, columns=azdias_clean.columns)
components_200_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LNR</th>
      <th>AKT_DAT_KL</th>
      <th>ALTERSKATEGORIE_FEIN</th>
      <th>ANZ_HAUSHALTE_AKTIV</th>
      <th>ANZ_HH_TITEL</th>
      <th>ANZ_KINDER</th>
      <th>ANZ_PERSONEN</th>
      <th>ANZ_STATISTISCHE_HAUSHALTE</th>
      <th>ANZ_TITEL</th>
      <th>ARBEIT</th>
      <th>...</th>
      <th>VHN</th>
      <th>VK_DHT4A</th>
      <th>VK_DISTANZ</th>
      <th>VK_ZG11</th>
      <th>W_KEIT_KIND_HH</th>
      <th>WOHNDAUER_2008</th>
      <th>WOHNLAGE</th>
      <th>ZABEOTYP</th>
      <th>ANREDE_KZ</th>
      <th>ALTERSKATEGORIE_GROB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.025196</td>
      <td>0.068999</td>
      <td>0.036918</td>
      <td>0.083827</td>
      <td>0.019283</td>
      <td>-0.023839</td>
      <td>-0.079215</td>
      <td>0.082442</td>
      <td>-0.003574</td>
      <td>0.097980</td>
      <td>...</td>
      <td>0.037251</td>
      <td>0.082931</td>
      <td>0.091584</td>
      <td>0.088679</td>
      <td>0.048126</td>
      <td>-0.037147</td>
      <td>-0.051684</td>
      <td>0.091885</td>
      <td>0.006827</td>
      <td>-0.047700</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.082493</td>
      <td>0.012866</td>
      <td>-0.003847</td>
      <td>0.017996</td>
      <td>0.025954</td>
      <td>-0.007078</td>
      <td>-0.018087</td>
      <td>0.017058</td>
      <td>0.010781</td>
      <td>-0.043246</td>
      <td>...</td>
      <td>0.041022</td>
      <td>0.022211</td>
      <td>0.025272</td>
      <td>0.027036</td>
      <td>0.016067</td>
      <td>-0.008524</td>
      <td>-0.081884</td>
      <td>-0.009615</td>
      <td>0.004982</td>
      <td>-0.002063</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.026103</td>
      <td>0.043504</td>
      <td>0.169462</td>
      <td>-0.019428</td>
      <td>-0.016311</td>
      <td>0.077602</td>
      <td>0.059547</td>
      <td>-0.022219</td>
      <td>-0.007254</td>
      <td>-0.042935</td>
      <td>...</td>
      <td>-0.017053</td>
      <td>-0.084706</td>
      <td>-0.027540</td>
      <td>0.027882</td>
      <td>-0.121000</td>
      <td>-0.064308</td>
      <td>0.028806</td>
      <td>0.044998</td>
      <td>0.010338</td>
      <td>-0.190105</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.048692</td>
      <td>-0.102890</td>
      <td>0.001029</td>
      <td>0.024576</td>
      <td>0.015947</td>
      <td>0.058542</td>
      <td>0.150540</td>
      <td>0.024358</td>
      <td>0.011628</td>
      <td>0.084724</td>
      <td>...</td>
      <td>0.011933</td>
      <td>-0.186808</td>
      <td>-0.195728</td>
      <td>-0.170930</td>
      <td>-0.106422</td>
      <td>0.012627</td>
      <td>-0.074840</td>
      <td>-0.047315</td>
      <td>-0.004984</td>
      <td>0.011675</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.114306</td>
      <td>0.034305</td>
      <td>0.026828</td>
      <td>0.009713</td>
      <td>0.009655</td>
      <td>-0.010156</td>
      <td>-0.043366</td>
      <td>0.012304</td>
      <td>0.000426</td>
      <td>0.072168</td>
      <td>...</td>
      <td>-0.030803</td>
      <td>0.027714</td>
      <td>0.024074</td>
      <td>0.015732</td>
      <td>0.015764</td>
      <td>-0.019528</td>
      <td>-0.015551</td>
      <td>0.021424</td>
      <td>0.001547</td>
      <td>-0.039737</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 307 columns</p>
</div>




```python
#list comprehension to create the index
comp = ['comp_' + str(i)for i in components_200_df.index + 1]
comp
```




    ['comp_1',
     'comp_2',
     'comp_3',
     'comp_4',
     'comp_5',
     'comp_6',
     'comp_7',
     'comp_8',
     'comp_9',
     'comp_10',
     'comp_11',
     'comp_12',
     'comp_13',
     'comp_14',
     'comp_15',
     'comp_16',
     'comp_17',
     'comp_18',
     'comp_19',
     'comp_20',
     'comp_21',
     'comp_22',
     'comp_23',
     'comp_24',
     'comp_25',
     'comp_26',
     'comp_27',
     'comp_28',
     'comp_29',
     'comp_30',
     'comp_31',
     'comp_32',
     'comp_33',
     'comp_34',
     'comp_35',
     'comp_36',
     'comp_37',
     'comp_38',
     'comp_39',
     'comp_40',
     'comp_41',
     'comp_42',
     'comp_43',
     'comp_44',
     'comp_45',
     'comp_46',
     'comp_47',
     'comp_48',
     'comp_49',
     'comp_50',
     'comp_51',
     'comp_52',
     'comp_53',
     'comp_54',
     'comp_55',
     'comp_56',
     'comp_57',
     'comp_58',
     'comp_59',
     'comp_60',
     'comp_61',
     'comp_62',
     'comp_63',
     'comp_64',
     'comp_65',
     'comp_66',
     'comp_67',
     'comp_68',
     'comp_69',
     'comp_70',
     'comp_71',
     'comp_72',
     'comp_73',
     'comp_74',
     'comp_75',
     'comp_76',
     'comp_77',
     'comp_78',
     'comp_79',
     'comp_80',
     'comp_81',
     'comp_82',
     'comp_83',
     'comp_84',
     'comp_85',
     'comp_86',
     'comp_87',
     'comp_88',
     'comp_89',
     'comp_90',
     'comp_91',
     'comp_92',
     'comp_93',
     'comp_94',
     'comp_95',
     'comp_96',
     'comp_97',
     'comp_98',
     'comp_99',
     'comp_100',
     'comp_101',
     'comp_102',
     'comp_103',
     'comp_104',
     'comp_105',
     'comp_106',
     'comp_107',
     'comp_108',
     'comp_109',
     'comp_110',
     'comp_111',
     'comp_112',
     'comp_113',
     'comp_114',
     'comp_115',
     'comp_116',
     'comp_117',
     'comp_118',
     'comp_119',
     'comp_120',
     'comp_121',
     'comp_122',
     'comp_123',
     'comp_124',
     'comp_125',
     'comp_126',
     'comp_127',
     'comp_128',
     'comp_129',
     'comp_130',
     'comp_131',
     'comp_132',
     'comp_133',
     'comp_134',
     'comp_135',
     'comp_136',
     'comp_137',
     'comp_138',
     'comp_139',
     'comp_140',
     'comp_141',
     'comp_142',
     'comp_143',
     'comp_144',
     'comp_145',
     'comp_146',
     'comp_147',
     'comp_148',
     'comp_149',
     'comp_150',
     'comp_151',
     'comp_152',
     'comp_153',
     'comp_154',
     'comp_155',
     'comp_156',
     'comp_157',
     'comp_158',
     'comp_159',
     'comp_160',
     'comp_161',
     'comp_162',
     'comp_163',
     'comp_164',
     'comp_165',
     'comp_166',
     'comp_167',
     'comp_168',
     'comp_169',
     'comp_170',
     'comp_171',
     'comp_172',
     'comp_173',
     'comp_174',
     'comp_175',
     'comp_176',
     'comp_177',
     'comp_178',
     'comp_179',
     'comp_180',
     'comp_181',
     'comp_182',
     'comp_183',
     'comp_184',
     'comp_185',
     'comp_186',
     'comp_187',
     'comp_188',
     'comp_189',
     'comp_190',
     'comp_191',
     'comp_192',
     'comp_193',
     'comp_194',
     'comp_195',
     'comp_196',
     'comp_197',
     'comp_198',
     'comp_199',
     'comp_200']




```python
components_200_df.shape
```




    (200, 307)



At this stage i will create a DataFrame from the returned PCA object, after creating the DF then we can visualize the weights of each component and see the correlations between the features and if the provide any insight


```python
components_200_df.index = comp
```


```python
components_200_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LNR</th>
      <th>AKT_DAT_KL</th>
      <th>ALTERSKATEGORIE_FEIN</th>
      <th>ANZ_HAUSHALTE_AKTIV</th>
      <th>ANZ_HH_TITEL</th>
      <th>ANZ_KINDER</th>
      <th>ANZ_PERSONEN</th>
      <th>ANZ_STATISTISCHE_HAUSHALTE</th>
      <th>ANZ_TITEL</th>
      <th>ARBEIT</th>
      <th>...</th>
      <th>VHN</th>
      <th>VK_DHT4A</th>
      <th>VK_DISTANZ</th>
      <th>VK_ZG11</th>
      <th>W_KEIT_KIND_HH</th>
      <th>WOHNDAUER_2008</th>
      <th>WOHNLAGE</th>
      <th>ZABEOTYP</th>
      <th>ANREDE_KZ</th>
      <th>ALTERSKATEGORIE_GROB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>comp_1</th>
      <td>-0.025196</td>
      <td>0.068999</td>
      <td>0.036918</td>
      <td>0.083827</td>
      <td>0.019283</td>
      <td>-0.023839</td>
      <td>-0.079215</td>
      <td>0.082442</td>
      <td>-0.003574</td>
      <td>0.097980</td>
      <td>...</td>
      <td>0.037251</td>
      <td>0.082931</td>
      <td>0.091584</td>
      <td>0.088679</td>
      <td>0.048126</td>
      <td>-0.037147</td>
      <td>-0.051684</td>
      <td>0.091885</td>
      <td>0.006827</td>
      <td>-0.047700</td>
    </tr>
    <tr>
      <th>comp_2</th>
      <td>0.082493</td>
      <td>0.012866</td>
      <td>-0.003847</td>
      <td>0.017996</td>
      <td>0.025954</td>
      <td>-0.007078</td>
      <td>-0.018087</td>
      <td>0.017058</td>
      <td>0.010781</td>
      <td>-0.043246</td>
      <td>...</td>
      <td>0.041022</td>
      <td>0.022211</td>
      <td>0.025272</td>
      <td>0.027036</td>
      <td>0.016067</td>
      <td>-0.008524</td>
      <td>-0.081884</td>
      <td>-0.009615</td>
      <td>0.004982</td>
      <td>-0.002063</td>
    </tr>
    <tr>
      <th>comp_3</th>
      <td>0.026103</td>
      <td>0.043504</td>
      <td>0.169462</td>
      <td>-0.019428</td>
      <td>-0.016311</td>
      <td>0.077602</td>
      <td>0.059547</td>
      <td>-0.022219</td>
      <td>-0.007254</td>
      <td>-0.042935</td>
      <td>...</td>
      <td>-0.017053</td>
      <td>-0.084706</td>
      <td>-0.027540</td>
      <td>0.027882</td>
      <td>-0.121000</td>
      <td>-0.064308</td>
      <td>0.028806</td>
      <td>0.044998</td>
      <td>0.010338</td>
      <td>-0.190105</td>
    </tr>
    <tr>
      <th>comp_4</th>
      <td>-0.048692</td>
      <td>-0.102890</td>
      <td>0.001029</td>
      <td>0.024576</td>
      <td>0.015947</td>
      <td>0.058542</td>
      <td>0.150540</td>
      <td>0.024358</td>
      <td>0.011628</td>
      <td>0.084724</td>
      <td>...</td>
      <td>0.011933</td>
      <td>-0.186808</td>
      <td>-0.195728</td>
      <td>-0.170930</td>
      <td>-0.106422</td>
      <td>0.012627</td>
      <td>-0.074840</td>
      <td>-0.047315</td>
      <td>-0.004984</td>
      <td>0.011675</td>
    </tr>
    <tr>
      <th>comp_5</th>
      <td>-0.114306</td>
      <td>0.034305</td>
      <td>0.026828</td>
      <td>0.009713</td>
      <td>0.009655</td>
      <td>-0.010156</td>
      <td>-0.043366</td>
      <td>0.012304</td>
      <td>0.000426</td>
      <td>0.072168</td>
      <td>...</td>
      <td>-0.030803</td>
      <td>0.027714</td>
      <td>0.024074</td>
      <td>0.015732</td>
      <td>0.015764</td>
      <td>-0.019528</td>
      <td>-0.015551</td>
      <td>0.021424</td>
      <td>0.001547</td>
      <td>-0.039737</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 307 columns</p>
</div>




```python
feature_names_comp = components_200_df.columns
feature_names_comp
```




    Index(['LNR', 'AKT_DAT_KL', 'ALTERSKATEGORIE_FEIN', 'ANZ_HAUSHALTE_AKTIV',
           'ANZ_HH_TITEL', 'ANZ_KINDER', 'ANZ_PERSONEN',
           'ANZ_STATISTISCHE_HAUSHALTE', 'ANZ_TITEL', 'ARBEIT',
           ...
           'VHN', 'VK_DHT4A', 'VK_DISTANZ', 'VK_ZG11', 'W_KEIT_KIND_HH',
           'WOHNDAUER_2008', 'WOHNLAGE', 'ZABEOTYP', 'ANREDE_KZ',
           'ALTERSKATEGORIE_GROB'],
          dtype='object', length=307)



after completing the components DF we will create a function to visualize the weights and see how which features correlate

<b> Explaining each component </b> and what are the attributes within, and how do they correlate or not , together.


```python
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
    
    
```


```python
display_comp2(components_200_df, 0, 10)
```


![png](output_82_0.png)


          weights        features       abs
    257 -0.151102      MOBI_REGIO  0.151102
    133 -0.149552     KBA13_ANTG1  0.149552
    261 -0.149124      PLZ8_ANTG1  0.149124
    135  0.147629     KBA13_ANTG3  0.147629
    140  0.145534    KBA13_BAUMAX  0.145534
    136  0.143933     KBA13_ANTG4  0.143933
    264  0.143671     PLZ8_BAUMAX  0.143671
    253 -0.143516  LP_STATUS_FEIN  0.143516
    254 -0.141622  LP_STATUS_GROB  0.141622
    256 -0.138921     MOBI_RASTER  0.138921
    

The 1st component is related to homes with low households 1 to 2 (PLZ8_ANTG1)
and not correlated to most common type of homes (PLZ8_BAUMAX) but correlated 
with share of cars per household (KBA13_AUTOQUOTE)



```python
display_comp2(components_200_df, 1, 10)
```


![png](output_84_0.png)


          weights                     features       abs
    179  0.191145         KBA13_HERST_BMW_BENZ  0.191145
    231  0.174800         KBA13_SEG_SPORTWAGEN  0.174800
    214  0.164341               KBA13_MERCEDES  0.164341
    228  0.163714  KBA13_SEG_OBEREMITTELKLASSE  0.163714
    147  0.158706                    KBA13_BMW  0.158706
    236 -0.155806                KBA13_SITZE_5  0.155806
    204  0.154966                 KBA13_KW_121  0.154966
    235  0.149667                KBA13_SITZE_4  0.149667
    158  0.145484               KBA13_CCM_2501  0.145484
    190  0.144454                KBA13_KMH_211  0.144454
    

its seems that this component has many things to do with high end expensive cars that are related with each other like BMW, MERC, and sport car
but are not correlated with cars that have 5 seats(KBA13_SITZE_5)


```python
display_comp2(components_200_df, 2, 10)
```


![png](output_86_0.png)


          weights               features       abs
    13   0.209425              CJT_TYP_1  0.209425
    67   0.204862          FINANZ_SPARER  0.204862
    14   0.204466              CJT_TYP_2  0.204466
    69  -0.192939       FINANZ_VORSORGER  0.192939
    17  -0.192604              CJT_TYP_5  0.192604
    306 -0.190105   ALTERSKATEGORIE_GROB  0.190105
    16  -0.186488              CJT_TYP_4  0.186488
    68   0.186234  FINANZ_UNAUFFAELLIGER  0.186234
    282  0.182757          SEMIO_PFLICHT  0.182757
    64   0.180690         FINANZ_ANLEGER  0.180690
    

here the money saver (FINANZ_SPARER) and the unremarkable are highly
correlated (FINANZ_UNAUFFAELLIGER) and opposite them is the prepared(FINANZ_VORSORGER)


```python
display_comp2(components_200_df, 3, 10)
```


![png](output_88_0.png)


          weights             features       abs
    299 -0.195728           VK_DISTANZ  0.195728
    298 -0.186808             VK_DHT4A  0.186808
    36  -0.171965        D19_KONSUMTYP  0.171965
    300 -0.170930              VK_ZG11  0.170930
    37  -0.166901    D19_KONSUMTYP_MAX  0.166901
    250  0.157372      LP_FAMILIE_GROB  0.157372
    252  0.154681  LP_LEBENSPHASE_GROB  0.154681
    249  0.154185      LP_FAMILIE_FEIN  0.154185
    251  0.151981  LP_LEBENSPHASE_FEIN  0.151981
    6    0.150540         ANZ_PERSONEN  0.150540
    

<b> KMEANS and clustering our PCA data</b>

Before staring a K means cluster, we need to find out what is the best cluster number and to do that we will perform the elbow method, and by plotting the results of the SSE, we can then choose the least SSE that makes sense to then cluster the kmeans with.



```python
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
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    


![png](output_90_1.png)



After plotting, we can see that there is no clear elbow cutoff to choose from but we can see that after number 7 things start becoming linear. We can choose 7 cluster and test it on the data.

After choosing the number 7, we can fit the data now and perform clustering on the general population data



We start by creating a Kmeans object from sklearn, then choose the 7 cluster based on the plot above. We then fit the object to our PCA data and then re-run the kmeans using the predict. After fitting the kmeans object to population data then we use the predict function to come up with the cluster for the population data.


```python
%%time

azdias_kmeans = KMeans(7)
azdias_model = azdias_kmeans.fit(azdias_pca_200)# why did we fit here only
azdias_labels = azdias_model.predict(azdias_pca_200)# why we ran the file again here in the predict? Osama
```

    CPU times: user 8min 47s, sys: 26 s, total: 9min 13s
    Wall time: 10min 34s
    


```python
#a function to see the top weighst of each cluster
def explain_comp(cluster, n_weight):
    over_rep = pd.DataFrame.from_dict(dict(zip(azdias_clean.columns, 
    pca.inverse_transform(azdias_model.cluster_centers_[cluster]))), orient='index').rename(columns={0: 'feature_values'}).sort_values('feature_values', ascending=False)
    pd.concat((over_rep['feature_values'][:n_weight], over_rep['feature_values'][-n_weight:]), axis=0).plot(kind='barh')
```


```python
explain_comp(0, 10)
```


![png](output_95_0.png)



```python
explain_comp(5, 10)
```


![png](output_96_0.png)



```python
explain_comp(1, 10)
```


![png](output_97_0.png)



```python
explain_comp(6,10)
```


![png](output_98_0.png)


Then we get the customer cleaned file and pass it to the pca object we created for the population file, and get back our customer PCA data.
we then use the newly formed kmeans object based on the population data and pass it to the predict function of kmeans and get the labels of predicted clusters for the customer data


```python
%%time
customer_pca_200 = pca.transform(customers_clean) #why did we use only tranform here and not fit_transform? osama
```

    CPU times: user 1.35 s, sys: 996 ms, total: 2.34 s
    Wall time: 13 s
    


```python
customer_pca_200.shape
```




    (134197, 200)




```python
customer_labels = azdias_kmeans.predict(customer_pca_200)
```


```python
print(azdias_final.shape)
print(customers_final.shape)
print(azdias_labels.shape)
print(customer_labels.shape)
print(azdias_pca_200.shape)
print(customer_pca_200.shape)
```

<b>We now create a DataFrame for the azdias_pca_200 clusters with labels </b>


```python
azdias_cluster_df = pd.DataFrame(azdias_pca_200, columns=comp)
azdias_cluster_df['cluster_labels'] = azdias_labels
azdias_cluster_df.tail()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comp_1</th>
      <th>comp_2</th>
      <th>comp_3</th>
      <th>comp_4</th>
      <th>comp_5</th>
      <th>comp_6</th>
      <th>comp_7</th>
      <th>comp_8</th>
      <th>comp_9</th>
      <th>comp_10</th>
      <th>...</th>
      <th>comp_192</th>
      <th>comp_193</th>
      <th>comp_194</th>
      <th>comp_195</th>
      <th>comp_196</th>
      <th>comp_197</th>
      <th>comp_198</th>
      <th>comp_199</th>
      <th>comp_200</th>
      <th>cluster_labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>736961</th>
      <td>-0.687566</td>
      <td>-4.016342</td>
      <td>3.890719</td>
      <td>0.326674</td>
      <td>1.730512</td>
      <td>-4.464654</td>
      <td>2.181508</td>
      <td>6.076504</td>
      <td>-0.384224</td>
      <td>2.202983</td>
      <td>...</td>
      <td>0.755386</td>
      <td>-0.477785</td>
      <td>0.665155</td>
      <td>0.224624</td>
      <td>-0.127145</td>
      <td>-0.583461</td>
      <td>0.030754</td>
      <td>-0.430369</td>
      <td>-0.747274</td>
      <td>1</td>
    </tr>
    <tr>
      <th>736962</th>
      <td>8.314587</td>
      <td>1.694539</td>
      <td>-0.002113</td>
      <td>-0.105196</td>
      <td>-0.959232</td>
      <td>1.434235</td>
      <td>5.428950</td>
      <td>-1.021177</td>
      <td>-0.877866</td>
      <td>-0.447357</td>
      <td>...</td>
      <td>-0.218119</td>
      <td>0.485584</td>
      <td>0.096762</td>
      <td>0.071434</td>
      <td>-0.195711</td>
      <td>-0.090184</td>
      <td>0.373632</td>
      <td>-0.078557</td>
      <td>0.716272</td>
      <td>6</td>
    </tr>
    <tr>
      <th>736963</th>
      <td>-2.742566</td>
      <td>0.170327</td>
      <td>6.417238</td>
      <td>-2.604799</td>
      <td>0.974180</td>
      <td>-1.775603</td>
      <td>3.405224</td>
      <td>-1.174877</td>
      <td>-1.575027</td>
      <td>0.237972</td>
      <td>...</td>
      <td>-0.164519</td>
      <td>0.093442</td>
      <td>0.711567</td>
      <td>-0.393336</td>
      <td>-0.229286</td>
      <td>-0.229202</td>
      <td>-0.072556</td>
      <td>-0.292457</td>
      <td>0.021065</td>
      <td>1</td>
    </tr>
    <tr>
      <th>736964</th>
      <td>7.209220</td>
      <td>-4.829202</td>
      <td>4.003730</td>
      <td>1.608461</td>
      <td>-3.482049</td>
      <td>3.747022</td>
      <td>1.803589</td>
      <td>3.542442</td>
      <td>-0.758662</td>
      <td>-1.940544</td>
      <td>...</td>
      <td>0.712929</td>
      <td>0.296702</td>
      <td>0.064549</td>
      <td>-0.421948</td>
      <td>-0.023098</td>
      <td>0.008550</td>
      <td>-0.717427</td>
      <td>0.746332</td>
      <td>-0.195512</td>
      <td>6</td>
    </tr>
    <tr>
      <th>736965</th>
      <td>-1.715196</td>
      <td>0.528842</td>
      <td>-3.320756</td>
      <td>-4.885304</td>
      <td>1.797693</td>
      <td>2.870914</td>
      <td>-2.707510</td>
      <td>-3.621920</td>
      <td>-0.560880</td>
      <td>-1.705818</td>
      <td>...</td>
      <td>-0.076543</td>
      <td>-0.944849</td>
      <td>0.265806</td>
      <td>-0.342206</td>
      <td>-0.278851</td>
      <td>-0.189220</td>
      <td>-0.268085</td>
      <td>0.251885</td>
      <td>-0.394307</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 201 columns</p>
</div>



<b>We also create a DataFrame for the customer_200 clusters with labels </b>


```python
customer_cluster_df = pd.DataFrame(customer_pca_200, columns=comp)
customer_cluster_df['cluster_labels'] = customer_labels
customer_cluster_df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comp_1</th>
      <th>comp_2</th>
      <th>comp_3</th>
      <th>comp_4</th>
      <th>comp_5</th>
      <th>comp_6</th>
      <th>comp_7</th>
      <th>comp_8</th>
      <th>comp_9</th>
      <th>comp_10</th>
      <th>...</th>
      <th>comp_192</th>
      <th>comp_193</th>
      <th>comp_194</th>
      <th>comp_195</th>
      <th>comp_196</th>
      <th>comp_197</th>
      <th>comp_198</th>
      <th>comp_199</th>
      <th>comp_200</th>
      <th>cluster_labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>134192</th>
      <td>-2.041627</td>
      <td>13.217253</td>
      <td>-5.167839</td>
      <td>2.401246</td>
      <td>1.479523</td>
      <td>3.439428</td>
      <td>-1.950820</td>
      <td>-0.030983</td>
      <td>-3.413535</td>
      <td>3.160772</td>
      <td>...</td>
      <td>-0.010418</td>
      <td>0.226594</td>
      <td>0.166751</td>
      <td>0.067828</td>
      <td>-0.637539</td>
      <td>0.240299</td>
      <td>0.357498</td>
      <td>0.509328</td>
      <td>-0.286309</td>
      <td>2</td>
    </tr>
    <tr>
      <th>134193</th>
      <td>-5.895409</td>
      <td>2.176056</td>
      <td>-4.546608</td>
      <td>-5.254981</td>
      <td>0.676609</td>
      <td>-1.857740</td>
      <td>1.201150</td>
      <td>-3.140360</td>
      <td>0.769313</td>
      <td>-2.282578</td>
      <td>...</td>
      <td>0.018138</td>
      <td>0.419599</td>
      <td>0.011692</td>
      <td>0.386050</td>
      <td>0.294413</td>
      <td>-0.052135</td>
      <td>0.767673</td>
      <td>-0.385267</td>
      <td>0.480790</td>
      <td>0</td>
    </tr>
    <tr>
      <th>134194</th>
      <td>-4.317898</td>
      <td>0.082840</td>
      <td>-5.720417</td>
      <td>2.722024</td>
      <td>-1.684382</td>
      <td>1.634712</td>
      <td>-0.434399</td>
      <td>1.418121</td>
      <td>-0.390462</td>
      <td>-0.901657</td>
      <td>...</td>
      <td>-0.246536</td>
      <td>0.455994</td>
      <td>0.054068</td>
      <td>-0.915440</td>
      <td>-0.365315</td>
      <td>-0.384539</td>
      <td>-0.363891</td>
      <td>1.142130</td>
      <td>0.785583</td>
      <td>0</td>
    </tr>
    <tr>
      <th>134195</th>
      <td>-1.210536</td>
      <td>-1.053312</td>
      <td>-0.888523</td>
      <td>4.721747</td>
      <td>-0.787584</td>
      <td>-3.099894</td>
      <td>0.500416</td>
      <td>-0.000933</td>
      <td>3.494565</td>
      <td>-0.630033</td>
      <td>...</td>
      <td>-0.666413</td>
      <td>-0.359588</td>
      <td>0.336576</td>
      <td>0.356228</td>
      <td>-0.262181</td>
      <td>0.350983</td>
      <td>0.095656</td>
      <td>-0.884320</td>
      <td>0.411187</td>
      <td>5</td>
    </tr>
    <tr>
      <th>134196</th>
      <td>-10.025222</td>
      <td>-0.910672</td>
      <td>0.540694</td>
      <td>-0.809938</td>
      <td>-0.762740</td>
      <td>3.782067</td>
      <td>0.650476</td>
      <td>-2.159396</td>
      <td>-0.004504</td>
      <td>0.223077</td>
      <td>...</td>
      <td>-0.146151</td>
      <td>0.112197</td>
      <td>-0.192756</td>
      <td>0.117309</td>
      <td>0.175428</td>
      <td>-0.692120</td>
      <td>-0.231014</td>
      <td>-0.087123</td>
      <td>0.649830</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 201 columns</p>
</div>




```python
customer_cluster_df.shape
```




    (134197, 201)




```python
azdias_cluster_df.shape
```




    (736966, 201)




```python
print(customer_cluster_df['cluster_labels'].value_counts())
print(azdias_cluster_df['cluster_labels'].value_counts())
```

    0    42480
    5    37936
    2    28298
    3    16664
    4     4357
    6     2983
    1     1479
    Name: cluster_labels, dtype: int64
    1    131114
    0    124826
    5    113262
    3    104894
    6    102136
    4     84099
    2     76635
    Name: cluster_labels, dtype: int64
    


```python
label_count_customer = customer_cluster_df['cluster_labels'].value_counts()
label_count_azdias = azdias_cluster_df['cluster_labels'].value_counts()
```


```python
plt.figure(figsize = (10,10))
plt.bar(label_count_customer.index, label_count_customer, label='customer_clusters')
plt.bar(label_count_azdias.index, label_count_azdias, width=0.5, label='general_pop_clusters')
plt.legend()
plt.title('Customer and Population cluster comparison')
plt.xlabel('Clusters')
plt.show()
```


![png](output_112_0.png)


<b> In this step will choose the highest weights of the features and take the most over represented and the least represented and compare the means of the highest feature weights</b>

Cluster 0 is the overrepresented, and i will reverse the PCA and scaler matrix back to its original numbers to better compare the means of the choosen features


```python
reverse_cluster_df_label_0 = customer_cluster_df[customer_cluster_df['cluster_labels'] == 0]
reverse_cluster_df_label_0.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comp_1</th>
      <th>comp_2</th>
      <th>comp_3</th>
      <th>comp_4</th>
      <th>comp_5</th>
      <th>comp_6</th>
      <th>comp_7</th>
      <th>comp_8</th>
      <th>comp_9</th>
      <th>comp_10</th>
      <th>...</th>
      <th>comp_192</th>
      <th>comp_193</th>
      <th>comp_194</th>
      <th>comp_195</th>
      <th>comp_196</th>
      <th>comp_197</th>
      <th>comp_198</th>
      <th>comp_199</th>
      <th>comp_200</th>
      <th>cluster_labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-9.476384</td>
      <td>3.687293</td>
      <td>-3.737349</td>
      <td>-1.389049</td>
      <td>-0.930399</td>
      <td>3.390522</td>
      <td>-0.652245</td>
      <td>-0.050367</td>
      <td>-0.850223</td>
      <td>-0.497954</td>
      <td>...</td>
      <td>-0.279800</td>
      <td>-0.075741</td>
      <td>0.730156</td>
      <td>-0.187557</td>
      <td>-0.008946</td>
      <td>-0.995373</td>
      <td>0.724482</td>
      <td>-0.145383</td>
      <td>0.079320</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-5.845971</td>
      <td>-0.628103</td>
      <td>-4.482307</td>
      <td>-2.052258</td>
      <td>4.149916</td>
      <td>3.611791</td>
      <td>0.042437</td>
      <td>-3.418126</td>
      <td>0.485797</td>
      <td>-1.589972</td>
      <td>...</td>
      <td>0.714102</td>
      <td>0.483934</td>
      <td>0.222817</td>
      <td>-0.542510</td>
      <td>0.451654</td>
      <td>0.791467</td>
      <td>0.476705</td>
      <td>0.047095</td>
      <td>-0.167527</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-9.236158</td>
      <td>1.108800</td>
      <td>-2.688727</td>
      <td>0.866361</td>
      <td>-2.517289</td>
      <td>3.081721</td>
      <td>-0.339249</td>
      <td>-2.100635</td>
      <td>-3.156163</td>
      <td>-1.892788</td>
      <td>...</td>
      <td>-0.440041</td>
      <td>-0.241670</td>
      <td>-0.251929</td>
      <td>0.852654</td>
      <td>-0.355661</td>
      <td>0.319946</td>
      <td>-0.044674</td>
      <td>-0.004794</td>
      <td>-0.722656</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-7.201212</td>
      <td>-0.916192</td>
      <td>-2.157627</td>
      <td>1.757947</td>
      <td>0.408651</td>
      <td>3.650375</td>
      <td>-0.778827</td>
      <td>0.606440</td>
      <td>2.286076</td>
      <td>1.138996</td>
      <td>...</td>
      <td>-0.344259</td>
      <td>0.003783</td>
      <td>-0.195984</td>
      <td>0.189320</td>
      <td>0.319783</td>
      <td>-0.280302</td>
      <td>0.415072</td>
      <td>0.486213</td>
      <td>0.872285</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-3.309725</td>
      <td>-0.831424</td>
      <td>-1.643436</td>
      <td>-0.452202</td>
      <td>2.484526</td>
      <td>-4.743645</td>
      <td>-0.656293</td>
      <td>2.545621</td>
      <td>1.283923</td>
      <td>0.154466</td>
      <td>...</td>
      <td>-1.237979</td>
      <td>1.305444</td>
      <td>0.535780</td>
      <td>1.025638</td>
      <td>-0.733555</td>
      <td>0.457018</td>
      <td>0.136811</td>
      <td>-0.925639</td>
      <td>0.345074</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 201 columns</p>
</div>




```python
reverse_cluster_df_label_0_droped = reverse_cluster_df_label_0.drop('cluster_labels', axis=1)
cluster_0_pca = pca.inverse_transform(reverse_cluster_df_label_0_droped)
cluster_0_scaler = scaler.inverse_transform(cluster_0_pca)
```


```python
cluster_0_final = pd.DataFrame(cluster_0_scaler, columns=customers_clean.columns)
cluster_0_final.shape
```




    (42480, 307)




```python
cluster_0_final.shape
```




    (42453, 307)




```python
azdias_model.cluster_centers_[0].shape
```




    (200,)




```python
azdias_kmeans
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=7, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0)




```python
cluster_0_final[imp_cols].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FINANZ_MINIMALIST</th>
      <th>CJT_TYP_5</th>
      <th>CJT_TYP_3</th>
      <th>LP_STATUS_GROB</th>
      <th>LP_STATUS_FEIN</th>
      <th>FINANZ_VORSORGER</th>
      <th>CJT_TYP_6</th>
      <th>KBA13_ANTG1</th>
      <th>PLZ8_ANTG1</th>
      <th>MOBI_REGIO</th>
      <th>...</th>
      <th>PLZ8_BAUMAX</th>
      <th>KBA13_BAUMAX</th>
      <th>ALTERSKATEGORIE_FEIN</th>
      <th>KBA13_ANTG4</th>
      <th>EWDICHTE</th>
      <th>ORTSGR_KLS9</th>
      <th>KBA13_ANTG3</th>
      <th>CJT_TYP_1</th>
      <th>FINANZ_SPARER</th>
      <th>CJT_TYP_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>...</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
      <td>42480.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.750477</td>
      <td>4.742715</td>
      <td>4.953602</td>
      <td>4.110050</td>
      <td>8.489968</td>
      <td>4.786053</td>
      <td>4.935595</td>
      <td>2.772119</td>
      <td>2.932212</td>
      <td>4.162203</td>
      <td>...</td>
      <td>1.041243</td>
      <td>1.087433</td>
      <td>8.668630</td>
      <td>0.247262</td>
      <td>3.113869</td>
      <td>4.145756</td>
      <td>0.953702</td>
      <td>1.633048</td>
      <td>1.039426</td>
      <td>1.273221</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.700402</td>
      <td>0.589951</td>
      <td>0.598700</td>
      <td>1.043719</td>
      <td>2.355126</td>
      <td>0.539422</td>
      <td>0.622633</td>
      <td>0.612941</td>
      <td>0.653774</td>
      <td>0.763610</td>
      <td>...</td>
      <td>0.371464</td>
      <td>0.397245</td>
      <td>3.634501</td>
      <td>0.318981</td>
      <td>1.291621</td>
      <td>1.727321</td>
      <td>0.671023</td>
      <td>0.609175</td>
      <td>0.476680</td>
      <td>0.520008</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.677220</td>
      <td>1.187058</td>
      <td>1.035857</td>
      <td>0.526616</td>
      <td>-0.098685</td>
      <td>0.779582</td>
      <td>0.800162</td>
      <td>0.662171</td>
      <td>0.762504</td>
      <td>0.715562</td>
      <td>...</td>
      <td>0.224763</td>
      <td>0.212683</td>
      <td>-1.984498</td>
      <td>-0.525018</td>
      <td>0.032913</td>
      <td>0.256395</td>
      <td>-0.848176</td>
      <td>0.567702</td>
      <td>0.134940</td>
      <td>0.284738</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.495634</td>
      <td>4.511796</td>
      <td>4.767488</td>
      <td>3.907146</td>
      <td>8.228797</td>
      <td>4.681288</td>
      <td>4.708394</td>
      <td>2.207001</td>
      <td>2.331707</td>
      <td>3.754841</td>
      <td>...</td>
      <td>0.854441</td>
      <td>0.884590</td>
      <td>7.600865</td>
      <td>0.014810</td>
      <td>2.151611</td>
      <td>2.620303</td>
      <td>0.444803</td>
      <td>1.173412</td>
      <td>0.777766</td>
      <td>0.934959</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.983329</td>
      <td>4.859335</td>
      <td>5.102079</td>
      <td>4.260382</td>
      <td>9.170817</td>
      <td>4.900594</td>
      <td>5.065574</td>
      <td>2.843405</td>
      <td>3.006986</td>
      <td>4.282822</td>
      <td>...</td>
      <td>0.988587</td>
      <td>1.027973</td>
      <td>9.135770</td>
      <td>0.184449</td>
      <td>3.236787</td>
      <td>4.252957</td>
      <td>0.896212</td>
      <td>1.489659</td>
      <td>0.924607</td>
      <td>1.136541</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.240775</td>
      <td>5.129893</td>
      <td>5.335151</td>
      <td>4.904347</td>
      <td>10.049447</td>
      <td>5.091712</td>
      <td>5.336047</td>
      <td>3.153184</td>
      <td>3.341275</td>
      <td>4.707990</td>
      <td>...</td>
      <td>1.136017</td>
      <td>1.188714</td>
      <td>10.580852</td>
      <td>0.449467</td>
      <td>4.151440</td>
      <td>5.277050</td>
      <td>1.449315</td>
      <td>1.950969</td>
      <td>1.108427</td>
      <td>1.459894</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.033937</td>
      <td>6.223181</td>
      <td>6.905376</td>
      <td>5.510879</td>
      <td>11.944515</td>
      <td>6.041757</td>
      <td>6.363069</td>
      <td>4.271898</td>
      <td>4.516717</td>
      <td>5.704636</td>
      <td>...</td>
      <td>4.916011</td>
      <td>5.123377</td>
      <td>26.276842</td>
      <td>2.294528</td>
      <td>6.152674</td>
      <td>9.460032</td>
      <td>3.033199</td>
      <td>5.299993</td>
      <td>4.730341</td>
      <td>4.526401</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 307 columns</p>
</div>




Now with the least represented cluster which is 2, we will do the same, reverse the PCA and scale back to the original choosen features and compare their means


```python
reverse_cluster_df_label_1 = customer_cluster_df[customer_cluster_df['cluster_labels'] == 1]
reverse_cluster_df_label_1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comp_1</th>
      <th>comp_2</th>
      <th>comp_3</th>
      <th>comp_4</th>
      <th>comp_5</th>
      <th>comp_6</th>
      <th>comp_7</th>
      <th>comp_8</th>
      <th>comp_9</th>
      <th>comp_10</th>
      <th>...</th>
      <th>comp_192</th>
      <th>comp_193</th>
      <th>comp_194</th>
      <th>comp_195</th>
      <th>comp_196</th>
      <th>comp_197</th>
      <th>comp_198</th>
      <th>comp_199</th>
      <th>comp_200</th>
      <th>cluster_labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>-2.674457</td>
      <td>2.556158</td>
      <td>3.540663</td>
      <td>-3.675220</td>
      <td>1.955771</td>
      <td>3.027644</td>
      <td>-0.132425</td>
      <td>2.795348</td>
      <td>-4.251356</td>
      <td>0.782837</td>
      <td>...</td>
      <td>0.063950</td>
      <td>0.247076</td>
      <td>0.131290</td>
      <td>-0.602665</td>
      <td>-0.068232</td>
      <td>-0.880095</td>
      <td>-0.515273</td>
      <td>0.386126</td>
      <td>0.348215</td>
      <td>1</td>
    </tr>
    <tr>
      <th>141</th>
      <td>-1.215982</td>
      <td>-2.549614</td>
      <td>0.797171</td>
      <td>0.071357</td>
      <td>-1.775102</td>
      <td>0.178198</td>
      <td>-0.276493</td>
      <td>4.514481</td>
      <td>-0.735563</td>
      <td>1.434250</td>
      <td>...</td>
      <td>0.185319</td>
      <td>0.588604</td>
      <td>-0.156621</td>
      <td>0.476905</td>
      <td>0.371717</td>
      <td>0.712719</td>
      <td>-0.760812</td>
      <td>-0.447264</td>
      <td>-0.135168</td>
      <td>1</td>
    </tr>
    <tr>
      <th>229</th>
      <td>-0.227801</td>
      <td>3.839493</td>
      <td>2.664958</td>
      <td>-0.660287</td>
      <td>2.408558</td>
      <td>-3.287251</td>
      <td>-1.699484</td>
      <td>0.761873</td>
      <td>-0.321979</td>
      <td>-1.955530</td>
      <td>...</td>
      <td>-0.496809</td>
      <td>-1.113851</td>
      <td>0.013117</td>
      <td>-0.363956</td>
      <td>-0.316274</td>
      <td>-0.340689</td>
      <td>-0.645469</td>
      <td>-0.376886</td>
      <td>0.070895</td>
      <td>1</td>
    </tr>
    <tr>
      <th>274</th>
      <td>0.029414</td>
      <td>0.842304</td>
      <td>2.330542</td>
      <td>2.842876</td>
      <td>-0.838369</td>
      <td>3.414429</td>
      <td>-5.504154</td>
      <td>4.151050</td>
      <td>-1.994308</td>
      <td>-0.579562</td>
      <td>...</td>
      <td>0.936141</td>
      <td>1.152665</td>
      <td>-0.212486</td>
      <td>-0.214735</td>
      <td>0.754527</td>
      <td>0.124336</td>
      <td>-0.099831</td>
      <td>-0.408565</td>
      <td>0.520668</td>
      <td>1</td>
    </tr>
    <tr>
      <th>517</th>
      <td>0.480709</td>
      <td>1.999774</td>
      <td>-0.661685</td>
      <td>-1.919164</td>
      <td>1.830338</td>
      <td>1.095988</td>
      <td>-2.277618</td>
      <td>1.009475</td>
      <td>-0.947798</td>
      <td>-0.657429</td>
      <td>...</td>
      <td>-0.103923</td>
      <td>-0.226468</td>
      <td>0.390249</td>
      <td>-0.104793</td>
      <td>0.663871</td>
      <td>-0.125837</td>
      <td>0.173426</td>
      <td>0.777412</td>
      <td>0.263898</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 201 columns</p>
</div>




```python
reverse_cluster_df_label_1_droped = reverse_cluster_df_label_1.drop('cluster_labels', axis=1)
cluster_1_pca = pca.inverse_transform(reverse_cluster_df_label_1_droped)
cluster_1_scaler = scaler.inverse_transform(cluster_1_pca)
```


```python
cluster_1_scaler.shape
```




    (1479, 307)




```python
cluster_1_final = pd.DataFrame(cluster_1_scaler, columns=customers_clean.columns)
cluster_1_final.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LNR</th>
      <th>AKT_DAT_KL</th>
      <th>ALTERSKATEGORIE_FEIN</th>
      <th>ANZ_HAUSHALTE_AKTIV</th>
      <th>ANZ_HH_TITEL</th>
      <th>ANZ_KINDER</th>
      <th>ANZ_PERSONEN</th>
      <th>ANZ_STATISTISCHE_HAUSHALTE</th>
      <th>ANZ_TITEL</th>
      <th>ARBEIT</th>
      <th>...</th>
      <th>VHN</th>
      <th>VK_DHT4A</th>
      <th>VK_DISTANZ</th>
      <th>VK_ZG11</th>
      <th>W_KEIT_KIND_HH</th>
      <th>WOHNDAUER_2008</th>
      <th>WOHNLAGE</th>
      <th>ZABEOTYP</th>
      <th>ANREDE_KZ</th>
      <th>ALTERSKATEGORIE_GROB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>168815.474219</td>
      <td>7.281176</td>
      <td>9.554483</td>
      <td>1.043857</td>
      <td>-0.003268</td>
      <td>-0.005240</td>
      <td>1.008420</td>
      <td>1.299175</td>
      <td>-0.000170</td>
      <td>2.554420</td>
      <td>...</td>
      <td>0.947709</td>
      <td>6.500861</td>
      <td>8.642861</td>
      <td>6.676229</td>
      <td>4.014152</td>
      <td>3.818874</td>
      <td>2.594328</td>
      <td>0.821149</td>
      <td>0.934148</td>
      <td>1.636672</td>
    </tr>
    <tr>
      <th>1</th>
      <td>154249.233673</td>
      <td>1.287095</td>
      <td>16.015650</td>
      <td>0.360191</td>
      <td>0.010833</td>
      <td>-0.034885</td>
      <td>0.976879</td>
      <td>0.437728</td>
      <td>-0.000122</td>
      <td>3.784712</td>
      <td>...</td>
      <td>3.771821</td>
      <td>7.431041</td>
      <td>9.126550</td>
      <td>6.535232</td>
      <td>5.183955</td>
      <td>9.897192</td>
      <td>2.793572</td>
      <td>4.036501</td>
      <td>1.181706</td>
      <td>2.702302</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70203.374738</td>
      <td>6.518095</td>
      <td>17.202778</td>
      <td>11.046308</td>
      <td>0.002581</td>
      <td>-0.072225</td>
      <td>1.080638</td>
      <td>10.087988</td>
      <td>0.000169</td>
      <td>2.321837</td>
      <td>...</td>
      <td>1.935018</td>
      <td>4.734436</td>
      <td>7.097438</td>
      <td>6.881282</td>
      <td>2.752487</td>
      <td>3.711700</td>
      <td>2.399872</td>
      <td>6.001415</td>
      <td>2.043880</td>
      <td>1.991886</td>
    </tr>
    <tr>
      <th>3</th>
      <td>82781.755084</td>
      <td>4.614142</td>
      <td>13.657316</td>
      <td>4.260390</td>
      <td>0.000477</td>
      <td>-0.029295</td>
      <td>1.070205</td>
      <td>3.633231</td>
      <td>-0.000123</td>
      <td>3.868775</td>
      <td>...</td>
      <td>1.051098</td>
      <td>5.115584</td>
      <td>7.704938</td>
      <td>6.328392</td>
      <td>3.728345</td>
      <td>4.388946</td>
      <td>1.996004</td>
      <td>2.988287</td>
      <td>0.837090</td>
      <td>1.506189</td>
    </tr>
    <tr>
      <th>4</th>
      <td>177128.497740</td>
      <td>0.767134</td>
      <td>14.338471</td>
      <td>11.053382</td>
      <td>-0.007995</td>
      <td>-0.036215</td>
      <td>1.030850</td>
      <td>9.906658</td>
      <td>0.000697</td>
      <td>3.648113</td>
      <td>...</td>
      <td>1.969211</td>
      <td>7.699937</td>
      <td>9.967057</td>
      <td>7.595497</td>
      <td>3.838091</td>
      <td>9.020689</td>
      <td>2.356933</td>
      <td>3.041536</td>
      <td>1.280570</td>
      <td>2.664079</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 307 columns</p>
</div>




```python
cluster_1_final[imp_cols].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FINANZ_MINIMALIST</th>
      <th>CJT_TYP_5</th>
      <th>CJT_TYP_3</th>
      <th>LP_STATUS_GROB</th>
      <th>LP_STATUS_FEIN</th>
      <th>FINANZ_VORSORGER</th>
      <th>CJT_TYP_6</th>
      <th>KBA13_ANTG1</th>
      <th>PLZ8_ANTG1</th>
      <th>MOBI_REGIO</th>
      <th>...</th>
      <th>PLZ8_BAUMAX</th>
      <th>KBA13_BAUMAX</th>
      <th>ALTERSKATEGORIE_FEIN</th>
      <th>KBA13_ANTG4</th>
      <th>EWDICHTE</th>
      <th>ORTSGR_KLS9</th>
      <th>KBA13_ANTG3</th>
      <th>CJT_TYP_1</th>
      <th>FINANZ_SPARER</th>
      <th>CJT_TYP_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>...</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
      <td>1479.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.533565</td>
      <td>2.197617</td>
      <td>2.543244</td>
      <td>2.216145</td>
      <td>4.364982</td>
      <td>2.394751</td>
      <td>2.701659</td>
      <td>2.438171</td>
      <td>2.570271</td>
      <td>3.071802</td>
      <td>...</td>
      <td>1.224245</td>
      <td>1.299657</td>
      <td>15.236818</td>
      <td>0.454107</td>
      <td>3.689459</td>
      <td>4.855719</td>
      <td>1.391257</td>
      <td>4.313146</td>
      <td>3.663874</td>
      <td>4.006039</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.052515</td>
      <td>0.707727</td>
      <td>0.881877</td>
      <td>1.265868</td>
      <td>2.941645</td>
      <td>0.895304</td>
      <td>0.896076</td>
      <td>0.612148</td>
      <td>0.655747</td>
      <td>1.051722</td>
      <td>...</td>
      <td>0.651656</td>
      <td>0.688340</td>
      <td>4.944292</td>
      <td>0.379131</td>
      <td>1.215152</td>
      <td>1.711436</td>
      <td>0.682696</td>
      <td>0.720882</td>
      <td>0.884262</td>
      <td>0.707045</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.436098</td>
      <td>0.651397</td>
      <td>0.559052</td>
      <td>0.506755</td>
      <td>0.390132</td>
      <td>0.731618</td>
      <td>0.345022</td>
      <td>0.831804</td>
      <td>0.919111</td>
      <td>0.800184</td>
      <td>...</td>
      <td>0.294399</td>
      <td>0.272372</td>
      <td>-1.231559</td>
      <td>-0.420517</td>
      <td>0.090940</td>
      <td>0.660885</td>
      <td>-0.683465</td>
      <td>1.325302</td>
      <td>0.848351</td>
      <td>1.551006</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.741767</td>
      <td>1.679306</td>
      <td>1.917587</td>
      <td>1.143608</td>
      <td>1.855796</td>
      <td>1.722678</td>
      <td>2.010620</td>
      <td>1.953529</td>
      <td>2.047379</td>
      <td>2.263642</td>
      <td>...</td>
      <td>0.880594</td>
      <td>0.923437</td>
      <td>14.769606</td>
      <td>0.160569</td>
      <td>2.840579</td>
      <td>3.838680</td>
      <td>0.862661</td>
      <td>3.906462</td>
      <td>2.890753</td>
      <td>3.630338</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.441892</td>
      <td>2.094718</td>
      <td>2.401827</td>
      <td>1.927530</td>
      <td>3.676677</td>
      <td>2.129562</td>
      <td>2.683643</td>
      <td>2.271485</td>
      <td>2.393910</td>
      <td>3.095457</td>
      <td>...</td>
      <td>1.035005</td>
      <td>1.093536</td>
      <td>16.525944</td>
      <td>0.431432</td>
      <td>3.899418</td>
      <td>4.846246</td>
      <td>1.472102</td>
      <td>4.449018</td>
      <td>3.925706</td>
      <td>4.153692</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.231728</td>
      <td>2.695933</td>
      <td>3.001524</td>
      <td>2.929371</td>
      <td>6.131363</td>
      <td>3.174277</td>
      <td>3.317742</td>
      <td>2.911629</td>
      <td>3.077825</td>
      <td>3.920348</td>
      <td>...</td>
      <td>1.268904</td>
      <td>1.356934</td>
      <td>17.661557</td>
      <td>0.721586</td>
      <td>4.623407</td>
      <td>5.915817</td>
      <td>1.957162</td>
      <td>4.855166</td>
      <td>4.360416</td>
      <td>4.515774</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.249130</td>
      <td>5.492420</td>
      <td>5.920477</td>
      <td>5.384209</td>
      <td>11.564442</td>
      <td>5.596333</td>
      <td>5.507647</td>
      <td>4.289717</td>
      <td>4.497652</td>
      <td>5.456224</td>
      <td>...</td>
      <td>4.773354</td>
      <td>4.973014</td>
      <td>26.576124</td>
      <td>2.197430</td>
      <td>6.097201</td>
      <td>9.346727</td>
      <td>3.114264</td>
      <td>6.260483</td>
      <td>5.469711</td>
      <td>5.413330</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 307 columns</p>
</div>




```python
over_rep = cluster_0_final[imp_cols].describe().loc['mean']
over_rep

under_rep = cluster_1_final[imp_cols].describe().loc['mean']
under_rep
```




    FINANZ_MINIMALIST           2.533565
    CJT_TYP_5                   2.197617
    CJT_TYP_3                   2.543244
    LP_STATUS_GROB              2.216145
    LP_STATUS_FEIN              4.364982
    FINANZ_VORSORGER            2.394751
    CJT_TYP_6                   2.701659
    KBA13_ANTG1                 2.438171
    PLZ8_ANTG1                  2.570271
    MOBI_REGIO                  3.071802
    CJT_TYP_4                   2.188383
    MOBI_RASTER                 2.337121
    KBA13_AUTOQUOTE             3.054931
    GEMEINDETYP                26.347511
    ALTERSKATEGORIE_GROB        2.229805
    KBA05_GBZ                   3.207904
    KBA05_AUTOQUOT              3.227684
    KBA13_ALTERHALTER_60        3.035664
    KBA13_GBZ                   3.921270
    PLZ8_GBZ                    3.870432
    KBA13_HALTER_55             3.095825
    KBA13_HALTER_50             3.084312
    INNENSTADT                  5.348456
    KONSUMNAEHE                 3.005872
    KOMBIALTER                  2.348918
    CJT_KATALOGNUTZER           2.653046
    BALLRAUM                    4.496757
    FIRMENDICHTE                3.519639
    KBA13_SITZE_6               3.581050
    KBA13_SEG_GROSSRAUMVANS     3.515681
                                 ...    
    ZABEOTYP                    3.881218
    ONLINE_AFFINITAET           3.534685
    KBA13_ANTG2                 2.899432
    KBA13_HALTER_30             2.695751
    SEMIO_MAT                   4.549146
    PRAEGENDE_JUGENDJAHRE      12.046301
    STRUKTURTYP                 2.620674
    RELAT_AB                    2.748602
    RT_KEIN_ANREIZ              3.942958
    SEMIO_TRADV                 4.637188
    FINANZ_UNAUFFAELLIGER       3.576821
    SEMIO_REL                   4.939634
    SEMIO_RAT                   4.613283
    CJT_GESAMTTYP               4.030035
    HH_EINKOMMEN_SCORE          4.569836
    FINANZ_ANLEGER              3.617207
    ARBEIT                      2.807124
    PLZ8_ANTG3                  1.482775
    PLZ8_ANTG2                  2.844875
    SEMIO_PFLICHT               5.213966
    PLZ8_BAUMAX                 1.224245
    KBA13_BAUMAX                1.299657
    ALTERSKATEGORIE_FEIN       15.236818
    KBA13_ANTG4                 0.454107
    EWDICHTE                    3.689459
    ORTSGR_KLS9                 4.855719
    KBA13_ANTG3                 1.391257
    CJT_TYP_1                   4.313146
    FINANZ_SPARER               3.663874
    CJT_TYP_2                   4.006039
    Name: mean, Length: 307, dtype: float64




```python
clusters_2_and_mean = pd.concat([over_rep, under_rep], axis=1)
```


```python
clusters_2_and_mean.columns = ['Target', 'Not_Target']
```


```python
clusters_2_and_mean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>Not_Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FINANZ_MINIMALIST</th>
      <td>4.750477</td>
      <td>2.533565</td>
    </tr>
    <tr>
      <th>CJT_TYP_5</th>
      <td>4.742715</td>
      <td>2.197617</td>
    </tr>
    <tr>
      <th>CJT_TYP_3</th>
      <td>4.953602</td>
      <td>2.543244</td>
    </tr>
    <tr>
      <th>LP_STATUS_GROB</th>
      <td>4.110050</td>
      <td>2.216145</td>
    </tr>
    <tr>
      <th>LP_STATUS_FEIN</th>
      <td>8.489968</td>
      <td>4.364982</td>
    </tr>
  </tbody>
</table>
</div>



<b> Compare the means and analyse which features shows a high difference between the target group and the non-target group<b/>


```python
fig, ax = plt.subplots(figsize=(20,12))
plt.title("Mean Values of Columns Between Target and Non-Target Customers", fontsize=16)
plt.xlabel("Mean", fontsize=12)
clusters_2_and_mean.head().plot.bar(ax=ax)
plt.xticks(rotation=45)
```




    (array([0, 1, 2, 3, 4]), <a list of 5 Text xticklabel objects>)




![png](output_133_1.png)


<b>Conclusion</b>
We can see from the figure above that the target customers are high money savers: FINANZ_SPARER

financial typology: money saver	-1	unknown
	1	very high
	2	high
	3	average
	4	low
	5	very low


And the non-target is not and for the mobility features, our target should be people who are low in mobility: MOBI_REGIO
moving patterns	1	very high mobility
	2	high mobility
	3	middle mobility
	4	low mobility
	5	very low mobility 
	6	none



<b> Heatmap to better explain</b>

We will be using a Heatmap to better explain the correlation between the clusters and the components, and also prepare the data from the kmeans model. 



```python
azdias_model.cluster_centers_.shape
```




    (7, 200)




```python
plt.figure(figsize = (20,12))
ax = sns.heatmap(azdias_model.cluster_centers_.T[0:10], cmap = 'YlGnBu', annot=True)
ax.set_xlabel("Cluster")
ax.set_ylabel("Components")
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
ax.set_title("Attribute Value by Centroid")
plt.show()
```


```python
t = pca.inverse_transform(azdias_model.cluster_centers_)
```


```python
t2 = pd.DataFrame(t, columns=azdias_clean.columns)
t2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LNR</th>
      <th>AKT_DAT_KL</th>
      <th>ALTERSKATEGORIE_FEIN</th>
      <th>ANZ_HAUSHALTE_AKTIV</th>
      <th>ANZ_HH_TITEL</th>
      <th>ANZ_KINDER</th>
      <th>ANZ_PERSONEN</th>
      <th>ANZ_STATISTISCHE_HAUSHALTE</th>
      <th>ANZ_TITEL</th>
      <th>ARBEIT</th>
      <th>...</th>
      <th>VHN</th>
      <th>VK_DHT4A</th>
      <th>VK_DISTANZ</th>
      <th>VK_ZG11</th>
      <th>W_KEIT_KIND_HH</th>
      <th>WOHNDAUER_2008</th>
      <th>WOHNLAGE</th>
      <th>ZABEOTYP</th>
      <th>ANREDE_KZ</th>
      <th>ALTERSKATEGORIE_GROB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.212064</td>
      <td>-0.292544</td>
      <td>-0.660482</td>
      <td>-0.374500</td>
      <td>-0.096182</td>
      <td>-0.241423</td>
      <td>-0.189034</td>
      <td>-0.365314</td>
      <td>-0.010240</td>
      <td>-0.599152</td>
      <td>...</td>
      <td>-0.181065</td>
      <td>0.248179</td>
      <td>0.058562</td>
      <td>-0.135429</td>
      <td>0.409359</td>
      <td>0.230071</td>
      <td>0.382039</td>
      <td>-0.414096</td>
      <td>-0.092071</td>
      <td>0.656736</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.300517</td>
      <td>0.464848</td>
      <td>0.624548</td>
      <td>-0.093915</td>
      <td>-0.073722</td>
      <td>0.067149</td>
      <td>-0.249927</td>
      <td>-0.100415</td>
      <td>-0.044991</td>
      <td>-0.263378</td>
      <td>...</td>
      <td>-0.076275</td>
      <td>0.200460</td>
      <td>0.435406</td>
      <td>0.588248</td>
      <td>-0.061028</td>
      <td>-0.358146</td>
      <td>0.042936</td>
      <td>0.441642</td>
      <td>0.117059</td>
      <td>-0.782332</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.212207</td>
      <td>-0.083933</td>
      <td>-0.196647</td>
      <td>-0.050760</td>
      <td>0.212727</td>
      <td>-0.072112</td>
      <td>-0.063634</td>
      <td>-0.046484</td>
      <td>0.150315</td>
      <td>-0.090634</td>
      <td>...</td>
      <td>0.268284</td>
      <td>0.012060</td>
      <td>-0.066719</td>
      <td>-0.115802</td>
      <td>0.098136</td>
      <td>-0.000537</td>
      <td>-0.666576</td>
      <td>-0.236407</td>
      <td>-0.028338</td>
      <td>0.215142</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.209951</td>
      <td>0.000523</td>
      <td>-0.503022</td>
      <td>0.371151</td>
      <td>0.065229</td>
      <td>-0.245536</td>
      <td>-0.312318</td>
      <td>0.375902</td>
      <td>-0.007330</td>
      <td>0.432919</td>
      <td>...</td>
      <td>0.203890</td>
      <td>0.482216</td>
      <td>0.359369</td>
      <td>0.215226</td>
      <td>0.451251</td>
      <td>0.155832</td>
      <td>-0.328754</td>
      <td>0.076777</td>
      <td>0.052056</td>
      <td>0.605457</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.360997</td>
      <td>-0.009539</td>
      <td>-0.041284</td>
      <td>0.041903</td>
      <td>-0.040535</td>
      <td>-0.089995</td>
      <td>-0.135187</td>
      <td>0.061571</td>
      <td>-0.025110</td>
      <td>0.830419</td>
      <td>...</td>
      <td>-0.191253</td>
      <td>0.001083</td>
      <td>-0.138109</td>
      <td>-0.259298</td>
      <td>0.107036</td>
      <td>0.009120</td>
      <td>0.303603</td>
      <td>0.185398</td>
      <td>-0.015396</td>
      <td>0.087168</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.146274</td>
      <td>-0.508409</td>
      <td>0.217562</td>
      <td>-0.403835</td>
      <td>-0.099773</td>
      <td>0.567873</td>
      <td>1.200701</td>
      <td>-0.403642</td>
      <td>0.009280</td>
      <td>-0.486988</td>
      <td>...</td>
      <td>-0.246625</td>
      <td>-1.196190</td>
      <td>-1.105276</td>
      <td>-0.879319</td>
      <td>-1.037209</td>
      <td>0.308342</td>
      <td>0.268092</td>
      <td>-0.656285</td>
      <td>-0.020725</td>
      <td>-0.069686</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.061314</td>
      <td>0.394915</td>
      <td>0.462591</td>
      <td>0.648465</td>
      <td>0.129606</td>
      <td>-0.040355</td>
      <td>-0.299624</td>
      <td>0.621083</td>
      <td>-0.024593</td>
      <td>0.549998</td>
      <td>...</td>
      <td>0.339472</td>
      <td>0.260446</td>
      <td>0.389732</td>
      <td>0.464787</td>
      <td>0.102778</td>
      <td>-0.330545</td>
      <td>-0.231601</td>
      <td>0.612803</td>
      <td>-0.034255</td>
      <td>-0.576303</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 307 columns</p>
</div>




```python
plt.figure(figsize = (25,30))
sns.heatmap(t2.T, cmap = 'YlGnBu')
xlabel('Features')
ylabel("clusters")
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 10)
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-173-2a6a5ec3779d> in <module>()
          1 plt.figure(figsize = (25,30))
          2 sns.heatmap(t2.T, cmap = 'YlGnBu')
    ----> 3 xlabel('Features')
          4 ylabel("clusters")
          5 plt.yticks(fontsize = 20)
    

    NameError: name 'xlabel' is not defined



![png](output_140_1.png)


<b>The above heat map will help in interpreting some of the clusters:</b>

Based on our bar plots: The similar clusters are 4, 5, 1 and the dissimilar are 2,6,3



    
    
    
    


```python
display_comp2(components_200_df, 2, 10)
```


![png](output_142_0.png)


          weights               features       abs
    13   0.209425              CJT_TYP_1  0.209425
    67   0.204862          FINANZ_SPARER  0.204862
    14   0.204466              CJT_TYP_2  0.204466
    69  -0.192939       FINANZ_VORSORGER  0.192939
    17  -0.192604              CJT_TYP_5  0.192604
    306 -0.190105   ALTERSKATEGORIE_GROB  0.190105
    16  -0.186488              CJT_TYP_4  0.186488
    68   0.186234  FINANZ_UNAUFFAELLIGER  0.186234
    282  0.182757          SEMIO_PFLICHT  0.182757
    64   0.180690         FINANZ_ANLEGER  0.180690
    






```python
display_comp2(components_200_df, 1, 10)
```


![png](output_144_0.png)


          weights                     features       abs
    179  0.191145         KBA13_HERST_BMW_BENZ  0.191145
    231  0.174800         KBA13_SEG_SPORTWAGEN  0.174800
    214  0.164341               KBA13_MERCEDES  0.164341
    228  0.163714  KBA13_SEG_OBEREMITTELKLASSE  0.163714
    147  0.158706                    KBA13_BMW  0.158706
    236 -0.155806                KBA13_SITZE_5  0.155806
    204  0.154966                 KBA13_KW_121  0.154966
    235  0.149667                KBA13_SITZE_4  0.149667
    158  0.145484               KBA13_CCM_2501  0.145484
    190  0.144454                KBA13_KMH_211  0.144454
    


```python
display_comp2(components_200_df, 0, 10)
```


![png](output_145_0.png)


          weights        features       abs
    257 -0.151102      MOBI_REGIO  0.151102
    133 -0.149552     KBA13_ANTG1  0.149552
    261 -0.149124      PLZ8_ANTG1  0.149124
    135  0.147629     KBA13_ANTG3  0.147629
    140  0.145534    KBA13_BAUMAX  0.145534
    136  0.143933     KBA13_ANTG4  0.143933
    264  0.143671     PLZ8_BAUMAX  0.143671
    253 -0.143516  LP_STATUS_FEIN  0.143516
    254 -0.141622  LP_STATUS_GROB  0.141622
    256 -0.138921     MOBI_RASTER  0.138921
    


```python


```

## Part 2: Supervised Learning Model

Now that you've found which parts of the population are more likely to be customers of the mail-order company, it's time to build a prediction model. Each of the rows in the "MAILOUT" data files represents an individual that was targeted for a mailout campaign. Ideally, we should be able to use the demographic information from each individual to decide whether or not it will be worth it to include that person in the campaign.

The "MAILOUT" data has been split into two approximately equal parts, each with almost 43 000 data rows. In this part, you can verify your model with the "TRAIN" partition, which includes a column, "RESPONSE", that states whether or not a person became a customer of the company following the campaign. In the next part, you'll need to create predictions on the "TEST" partition, where the "RESPONSE" column has been withheld.


```python
mailout_train = pd.read_csv('../../data/Term2/capstone/arvato_data/Udacity_MAILOUT_052018_TRAIN.csv', sep=';')
```

    /opt/conda/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2785: DtypeWarning: Columns (18,19) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    


```python
mailout_train.shape
```




    (42962, 367)




```python
mailout_train.columns
```




    Index(['LNR', 'AGER_TYP', 'AKT_DAT_KL', 'ALTER_HH', 'ALTER_KIND1',
           'ALTER_KIND2', 'ALTER_KIND3', 'ALTER_KIND4', 'ALTERSKATEGORIE_FEIN',
           'ANZ_HAUSHALTE_AKTIV',
           ...
           'VK_DHT4A', 'VK_DISTANZ', 'VK_ZG11', 'W_KEIT_KIND_HH', 'WOHNDAUER_2008',
           'WOHNLAGE', 'ZABEOTYP', 'RESPONSE', 'ANREDE_KZ',
           'ALTERSKATEGORIE_GROB'],
          dtype='object', length=367)



<b> Imbalanced Response Vector</b>
    
When reviewing the RESPONSE variable we can see clearly that its imbalanced, the Positive is 1.2% only of the total. Based on such a response percentage a accuracy metric must be chosen accordingly. The best suited is the ROC. The ROC will show clearly the TP and TN ratios. If we used another metric these details won’t show. Due to the imbalance, the model will be biased.



```python
mailout_train['RESPONSE'].unique()
```




    array([0, 1])




```python
#checking the Response ratio
mailout_train['RESPONSE'].value_counts()
```




    0    42430
    1      532
    Name: RESPONSE, dtype: int64




```python
mailout_train['RESPONSE'].unique()
```




    array([0, 1])




```python
mailout_train['RESPONSE'].value_counts()
```




    0    42430
    1      532
    Name: RESPONSE, dtype: int64




```python
#to see if response variable is balanced or not

num_of_row = mailout_train.shape[0]
num_of_row
```




    42962




```python
#the response ratio is vert low and hence needs further metrics
(mailout_train[mailout_train['RESPONSE'] == 1].shape[0] / mailout_train.shape[0]) * 100
```




    1.2383036171500394



<b> Preprocess the data for Machine Learning</b>

mailout_train data needs to go through the preprocessing steps before fitting it to any model, i will use the same function used at the beginning of the exercise to clean the data.
Then remove any categorical columns, and separate the response variable. Then impute the missing elements and then finally to standardize the data.

After these steps we can proceed with testing out which classifier best fits the data.



```python
# highly imbalanced data set
sns.countplot(mailout_train['RESPONSE'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f31f8ba4908>




![png](output_159_1.png)



```python
mailout_train.shape
```




    (42962, 367)




```python
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
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>attribute</th>
      <th>missing_or_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGER_TYP</td>
      <td>[-1, 0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ALTERSKATEGORIE_GROB</td>
      <td>[-1,  0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ALTER_HH</td>
      <td>[0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ANREDE_KZ</td>
      <td>[-1,  0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BALLRAUM</td>
      <td>[-1]</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```


```python
#we must preprocess the data first
mailout_train_pre = preprocess_df_for_ML(mailout_train, excell_attibutes_com)
mailout_train_pre.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LNR</th>
      <th>AKT_DAT_KL</th>
      <th>ALTERSKATEGORIE_FEIN</th>
      <th>ANZ_HAUSHALTE_AKTIV</th>
      <th>ANZ_HH_TITEL</th>
      <th>ANZ_KINDER</th>
      <th>ANZ_PERSONEN</th>
      <th>ANZ_STATISTISCHE_HAUSHALTE</th>
      <th>ANZ_TITEL</th>
      <th>ARBEIT</th>
      <th>...</th>
      <th>VK_DHT4A</th>
      <th>VK_DISTANZ</th>
      <th>VK_ZG11</th>
      <th>W_KEIT_KIND_HH</th>
      <th>WOHNDAUER_2008</th>
      <th>WOHNLAGE</th>
      <th>ZABEOTYP</th>
      <th>RESPONSE</th>
      <th>ANREDE_KZ</th>
      <th>ALTERSKATEGORIE_GROB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1763.0</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1771.0</td>
      <td>4.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1776.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1460.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1783.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>53.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>44.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 314 columns</p>
</div>



<b>Remove the Categorical cols</b>


```python
cat_to_remove = mailout_train_pre.columns[mailout_train_pre.dtypes == 'object']
cat_to_remove
```




    Index(['CAMEO_DEU_2015', 'CAMEO_DEUG_2015', 'CAMEO_INTL_2015',
           'D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM', 'OST_WEST_KZ'],
          dtype='object')




```python
mailout_train_pre_nocat = mailout_train_pre.drop(cat_to_remove, axis=1)
```


```python
mailout_train_pre_nocat.drop('RESPONSE', axis=1, inplace=True)
```


```python
mailout_train_pre_nocat.shape
```




    (42962, 307)



<b> Create the Pipeline </b>

Start by creating the objects to impute and standardize the data, and then i create the pipeline object that will help in processing the files in better fashion.


```python
#ini imputer and scaler objects with default param
imputer  = Imputer()
scaler = StandardScaler()
```


```python
def create_piping_object():
    pipe = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler)
    ])
    
    return pipe
```


```python
piper = create_piping_object()
```

Pipeline object ready to be used to transform any Data Frame


```python
piper
```




    Pipeline(memory=None,
         steps=[('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)), ('scaler', StandardScaler(copy=True, with_mean=True, with_std=True))])




```python
mailout_ready_piped = piper.fit_transform(mailout_train_pre_nocat)
```

Convert the numpy matrix to DF for better readability


```python
mailout_ready_piped_df = pd.DataFrame(mailout_ready_piped, columns=mailout_train_pre_nocat.columns)
```


```python
mailout_ready_piped_df.shape
```




    (42962, 307)




```python
mailout_train_pre['RESPONSE'].head()
```




    0    0.0
    1    0.0
    2    0.0
    3    0.0
    4    0.0
    Name: RESPONSE, dtype: float64



<b> Model Performance </b>

Due to the response variable nature, a binary classifier will be used. I will create a function to run multiple models and see the best ROC metric, i will also use Cross Validation technique to help in generalization and not fitting to noise. The link below provides a clear explanation of CV.

The ROC metric was chose due to the imbalanced data set, we have only 1.2% positive (1) response. This is a very low number and we can’t use the accuracy metric to provide a clear score for the models.

https://medium.com/lumiata/cross-validation-for-imbalanced-datasets-9d203ba47e8

<b> Trying multiple Algorithms</b>

By creating a function that can test the ROC accuracy of three different classifiers, then based on the score dict, we can choose the best model to then optimize with hyper parameters.

The three classifiers to be used are:

AdaBoostClassifier - 
GradientBoostingClassifier - 
LogisticRegressionCV 


```python
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
```

After testing the three classifers, GradientBoostingClassifier shows ths highest ROC score at 75%


```python
%%time
run_algo_2(mailout_ready_piped_df, mailout_train['RESPONSE'])
```

    working on the algoAdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
              learning_rate=1.0, n_estimators=50, random_state=28)
    working on the algoGradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, n_estimators=100,
                  presort='auto', random_state=28, subsample=1.0, verbose=0,
                  warm_start=False)
    working on the algoLogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
               fit_intercept=True, intercept_scaling=1.0, max_iter=100,
               multi_class='ovr', n_jobs=1, penalty='l2', random_state=28,
               refit=True, scoring=None, solver='lbfgs', tol=0.0001, verbose=0)
    CPU times: user 7min 17s, sys: 1.27 s, total: 7min 18s
    Wall time: 7min 21s
    


```python
score_list.values()
```




    dict_values([0.7268052636655733, 0.75294545523609313, 0.61384470809701897])



<b> Hyper parameter evaluation</b>
At this stage we try different hyper parameters to increase our score a little further if possible



```python
#below is the params to test for and see the best predictor
param_grid = {'learning_rate':[0.1, 0.2],
             'n_estimators': [100],
             'max_depth' : [3,5],
              'min_samples_split': [2, 4]}
```


```python
%%time
grid = GridSearchCV(GradientBoostingClassifier(random_state=28), param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
```

    CPU times: user 194 µs, sys: 41 µs, total: 235 µs
    Wall time: 241 µs
    


```python
grid.fit(mailout_ready_piped_df, mailout_train['RESPONSE'])
```




    GridSearchCV(cv=5, error_score='raise',
           estimator=GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, n_estimators=100,
                  presort='auto', random_state=28, subsample=1.0, verbose=0,
                  warm_start=False),
           fit_params=None, iid=True, n_jobs=-1,
           param_grid={'learning_rate': [0.1, 0.2], 'n_estimators': [100], 'max_depth': [3, 5], 'min_samples_split': [2, 4]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='roc_auc', verbose=0)




```python
#to see all params that can be evaulated 
grid.best_estimator_
```




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=4,
                  min_weight_fraction_leaf=0.0, n_estimators=100,
                  presort='auto', random_state=28, subsample=1.0, verbose=0,
                  warm_start=False)




```python
grid.best_score_
```




    0.7534297767408884




```python
#these are the best param choosen by the model. These will be used later to ini the new model.
grid.best_params_
```




    {'learning_rate': 0.1,
     'max_depth': 3,
     'min_samples_split': 4,
     'n_estimators': 100}



<b> Model based on best parameters</b>

After trying the types of classifiers, the highest model result is the GradientBoostingClassifier at ROC score of 0.75, but it comes with some disadvantage, the speed of the fitting.

After choosing the model, i ran a grid search to come up with the best parameters. Hence to initiate the model with the new parameters.



```python
#ini the model with best param
GBC_tuned = GradientBoostingClassifier(learning_rate=0.1, 
                                       max_depth=3, min_samples_split=4,
                                      n_estimators=100)
```


```python
%%time
GBC_tuned.fit(mailout_ready_piped_df, mailout_train['RESPONSE'])
```

    CPU times: user 56.4 s, sys: 112 ms, total: 56.5 s
    Wall time: 57.2 s
    




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=4,
                  min_weight_fraction_leaf=0.0, n_estimators=100,
                  presort='auto', random_state=None, subsample=1.0, verbose=0,
                  warm_start=False)



## Part 3: Kaggle Competition

Now that you've created a model to predict which individuals are most likely to respond to a mailout campaign, it's time to test that model in competition through Kaggle. If you click on the link [here](http://www.kaggle.com/t/21e6d45d4c574c7fa2d868f0e8c83140), you'll be taken to the competition page where, if you have a Kaggle account, you can enter. If you're one of the top performers, you may have the chance to be contacted by a hiring manager from Arvato or Bertelsmann for an interview!

Your entry to the competition should be a CSV file with two columns. The first column should be a copy of "LNR", which acts as an ID number for each individual in the "TEST" partition. The second column, "RESPONSE", should be some measure of how likely each individual became a customer – this might not be a straightforward probability. As you should have found in Part 2, there is a large output class imbalance, where most individuals did not respond to the mailout. Thus, predicting individual classes and using accuracy does not seem to be an appropriate performance evaluation method. Instead, the competition will be using AUC to evaluate performance. The exact values of the "RESPONSE" column do not matter as much: only that the higher values try to capture as many of the actual customers as possible, early in the ROC curve sweep.


```python
mailout_test = pd.read_csv('../../data/Term2/capstone/arvato_data/Udacity_MAILOUT_052018_TEST.csv', sep=';')
```

    /opt/conda/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2785: DtypeWarning: Columns (18,19) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    

<b> Preprocess the testing data</b>

Now we have to clean and preprocess the mailout_test file in a similar fashion to the mailout_train. The difference here is there is no response variable.

After cleaning the mailout_test file, then we can use our model and predict the response variable.



```python
mailout_test.shape
```




    (42833, 366)




```python
#checking if the df has nans
mailout_test.isnull().sum().head()
```




    LNR                0
    AGER_TYP           0
    AKT_DAT_KL      6889
    ALTER_HH        6889
    ALTER_KIND1    40820
    dtype: int64




```python
mailout_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LNR</th>
      <th>AGER_TYP</th>
      <th>AKT_DAT_KL</th>
      <th>ALTER_HH</th>
      <th>ALTER_KIND1</th>
      <th>ALTER_KIND2</th>
      <th>ALTER_KIND3</th>
      <th>ALTER_KIND4</th>
      <th>ALTERSKATEGORIE_FEIN</th>
      <th>ANZ_HAUSHALTE_AKTIV</th>
      <th>...</th>
      <th>VHN</th>
      <th>VK_DHT4A</th>
      <th>VK_DISTANZ</th>
      <th>VK_ZG11</th>
      <th>W_KEIT_KIND_HH</th>
      <th>WOHNDAUER_2008</th>
      <th>WOHNLAGE</th>
      <th>ZABEOTYP</th>
      <th>ANREDE_KZ</th>
      <th>ALTERSKATEGORIE_GROB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1754</td>
      <td>2</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1770</td>
      <td>-1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1465</td>
      <td>2</td>
      <td>9.0</td>
      <td>16.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1470</td>
      <td>-1</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>2.0</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1478</td>
      <td>1</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 366 columns</p>
</div>




```python
#re reun the function to clean the file
mailout_test_pre = preprocess_df_for_ML(mailout_test, excell_attibutes_com)
```


```python
mailout_test_pre.shape
```




    (42833, 313)




```python
#Remove the catigorical features
cat_to_remove = mailout_test_pre.columns[mailout_test_pre.dtypes == 'object']
cat_to_remove
```




    Index(['CAMEO_DEU_2015', 'CAMEO_DEUG_2015', 'CAMEO_INTL_2015',
           'D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM', 'OST_WEST_KZ'],
          dtype='object')




```python
mailout_test_pre_no_cat = mailout_test_pre.drop(cat_to_remove, axis=1)
```


```python
#checking the shape of the df, to make sure its like the mailout_train
mailout_test_pre_no_cat.shape
```




    (42833, 307)




```python
print(mailout_test_pre.columns)
```

    Index(['LNR', 'AKT_DAT_KL', 'ALTERSKATEGORIE_FEIN', 'ANZ_HAUSHALTE_AKTIV',
           'ANZ_HH_TITEL', 'ANZ_KINDER', 'ANZ_PERSONEN',
           'ANZ_STATISTISCHE_HAUSHALTE', 'ANZ_TITEL', 'ARBEIT',
           ...
           'VHN', 'VK_DHT4A', 'VK_DISTANZ', 'VK_ZG11', 'W_KEIT_KIND_HH',
           'WOHNDAUER_2008', 'WOHNLAGE', 'ZABEOTYP', 'ANREDE_KZ',
           'ALTERSKATEGORIE_GROB'],
          dtype='object', length=313)
    


```python
#reuse the pipeline to impute and scale
mailout_test_piped = piper.fit_transform(mailout_test_pre_no_cat)
```


```python
mailout_test_piped.shape
```




    (42833, 307)




```python
#output is in a matrix numpy format
mailout_test_piped
```




    array([[-1.66587146, -0.32602263, -0.96263817, ...,  0.17844079,
            -1.21327567,  0.72869594],
           [-1.66522513, -0.32602263, -2.47361325, ...,  0.17844079,
            -1.21327567,  0.72869594],
           [-1.67754572,  4.70043489,  0.29650773, ...,  0.17844079,
             0.82421499,  0.72869594],
           ..., 
           [ 1.00781746, -0.32602263,  1.30382445, ..., -0.71392982,
            -1.21327567, -0.20610993],
           [ 1.00810023, -0.32602263, -2.47361325, ...,  0.17844079,
             0.82421499, -0.20610993],
           [ 1.04643544, -0.32602263,  1.05199527, ...,  0.17844079,
             0.82421499, -0.20610993]])



<b> Transformation </b>
Transform the imputed and scaled matrix to DF format


```python
mailout_test_piped_df = pd.DataFrame(mailout_test_piped, columns=mailout_test_pre_no_cat.columns)
```

Use the model to predict the responce variable based on our mailout_test file


```python
#using predict_proba to get a probabilty as a response
pred = GBC_tuned.predict_proba(mailout_test_piped_df)
```


```python
pred_Df = pd.DataFrame(index=mailout_test['LNR'], data=pred)
```


```python
pred_Df.rename(index=str, columns={0:'First', 1:'Response'}, inplace=True)
# so go for 0 or 1 as a response
```


```python
pred_Df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>First</th>
      <th>Response</th>
    </tr>
    <tr>
      <th>LNR</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1754.0</th>
      <td>0.956396</td>
      <td>0.043604</td>
    </tr>
    <tr>
      <th>1770.0</th>
      <td>0.982660</td>
      <td>0.017340</td>
    </tr>
    <tr>
      <th>1465.0</th>
      <td>0.988351</td>
      <td>0.011649</td>
    </tr>
    <tr>
      <th>1470.0</th>
      <td>0.997965</td>
      <td>0.002035</td>
    </tr>
    <tr>
      <th>1478.0</th>
      <td>0.997585</td>
      <td>0.002415</td>
    </tr>
  </tbody>
</table>
</div>




```python
pred_Df.drop('First', axis=1, inplace=True)
```


```python
#save file as csv for submission
for_submission = pred_Df.to_csv('kaggle_submit.csv', index=False)

```

<b> Feature Importance</b>

At this stage we can use the methods available in the model to see the most important features by weight 



```python
#check the len of features is the same as cols
len(GBC_tuned.feature_importances_)
```




    307




```python
#function to display the most important features of the classifier

def feature_display(clf):
    GBC_tuned_features = pd.DataFrame(clf.feature_importances_)
    idx = np.argsort(GBC_tuned.feature_importances_)[::-1]
    Features_df = pd.concat([GBC_tuned_features.iloc[idx], pd.Series(mailout_ready_piped_df.columns[idx])], axis=1)
    Features_df.columns = ['weights', 'feature_name']
    Features_df_10 = Features_df.iloc[idx][:10]
    Features_df_10.plot.barh(x = 'feature_name',
                    y = 'weights')
```


```python
feature_display(GBC_tuned) # evaulate the best split - read about these features when highest are what they carry as attrubites
```


![png](output_223_0.png)


<b> Summary</b>

The above plot shows that SEMIO_RAT is the largest weight the separates the data at best. Second comes the next in line, RETOURTYP_BK_S. These features with the highest weighs best explain how the tree is splitting the customer as a high response or low response. 



```python
def feature_plot(importances, X_train, y_train, num_feat=5):
     
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:num_feat]]
    values = importances[indices][:num_feat]
 
    # Create the plot
    fig = plt.figure(figsize = (16,9))
    plt.title("Normalized Weights for the Most Predictive Features", fontsize = 16)
    plt.barh(np.arange(num_feat), values[::-1], height = 0.6, align="center", \
          label = "Feature Weight")
    plt.barh(np.arange(num_feat) - 0.3, np.cumsum(values)[::-1], height = 0.2, align = "center", \
          label = "Cumulative Feature Weight")
    plt.yticks(np.arange(num_feat), columns[::-1])
    plt.xlabel("Weight", fontsize = 12)
    plt.ylabel('')
     
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.show()
```


```python
GradientBoostingClassifier.feature_importances_
```




    <property at 0x7f098f8e4d18>




```python
# finish the project report, export image to explain better on the report
```
