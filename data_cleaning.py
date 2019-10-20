import pandas as pd 
import numpy as np

def check_duplicates(df1):
    if df1.duplicated().sum():
        print("There were {} duplicates and they have been removed".format(df1.duplicated().sum()))
        df1.drop_duplicates(inplace=True)
    else:
        print("You are all clear of duplicates")
        
    return df1

def seasons(df2):
    
    df2.loc[:, 'month'] = df2['month'].map({'dec': 'winter', 'jan': 'winter', 'feb': 'winter',
                                               'mar': 'spring', 'apr': 'spring', 'may': 'spring',
                                               'jun': 'summer',  'jul': 'summer', 'aug': 'summer',
                                               'sep': 'fall', 'oct': 'fall', 'nov': 'fall'
                                               })
    return df2

def education(df3):
#     df3.loc[:, 'education']=df3['education'].map({'basic.9y': 'basic', 
#                                                       'baisc.6y': 'basic',
#                                                       'basic.4y': 'basic'
#                                                      })
    df3['education']=np.where(df3['education'] =='basic.6y', 'basic', df3['education'])
    df3['education']=np.where(df3['education'] =='basic.6y', 'basic', df3['education'])
    df3['education']=np.where(df3['education'] =='basic.4y', 'basic', df3['education'])
    
    return df3

def cleaning_data(df):
#     drop_duplicate = check_duplicates(df)
    convert_to_seasons = seasons(df)
    cleaned_data = education(convert_to_seasons)
    
    return cleaned_data
    
    
#      s.map({'cat': 'kitten', 'dog': 'puppy'})
# months = {
#     'jan': 'winter',
#     'feb': 'winter',
#     'may': 'spring', 
#     'jun': 'summer', 
#     'jul': 'summer', 
#     'aug': 'summer', 
#     'oct': 'fall', 
#     'nov': 'fall', 
#     'dec': 'winter', 
#     'mar': 'spring', 
#     'apr': 'spring',
#     'sep': 'fall'
# }
# def seasons(df):
#     df.loc[:,'month'] = df.loc[:,'month'].map({'dec': 'winter', 'jan': 'winter', 'feb': 'winter',
#                   'mar': 'spring', 'apr': 'spring', 'may': 'spring',
#                   'jun': 'summer',  'jul': 'summer', 'aug': 'summer',
#                   'sep': 'fall', 'oct': 'fall', 'nov': 'fall'
#                   })
#     return df