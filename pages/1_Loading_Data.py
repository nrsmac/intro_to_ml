import streamlit as st
import pandas as pd
import plotly.express as px


'''
# Intro: A first data science project

## Loading Data

Data is commonly stored in `.csv` files. This stand for 'comma-separated values'. It's simply a text file separated by commas. The first row is usually the column titles. Each row is separated by a new line. Here is an example of the raisins dataset we will be using. This file is in `data/raisins.csv`

```
,Area,MajorAxisLength,MinorAxisLength,Eccentricity,ConvexArea,Extent,Perimeter,Class
0,87524,442.2460114,253.291155,0.819738392,90546,0.758650579,1184.04,Kecimen
1,75166,406.690687,243.0324363,0.801805234,78789,0.68412957,1121.786,Kecimen
2,90856,442.2670483,266.3283177,0.798353619,93717,0.637612812,1208.575,Kecimen
```

It's not practical to use this file directly, so we will use the Pandas library to help us. We will store our `.csv` as a `DataFrame` object.

'''

df = pd.read_csv('data/raisins.csv')
df = df.drop('Unnamed: 0', axis=1)
df

'Examine the data. What are the dependent variables/predictors? What are we trying to predict (target)?'
if (st.checkbox('Show Answer')):
    """
>    __Predictors:__  
>	Area, 
>	MajorAxisLength,
>	MinorAxisLength,
>	Eccentricity,
>	ConvexArea,
>	Extent,
>	Perimeter
>
    """

    "> __Target__: Class"

"""
If you haven't figured it out yet, this dataset has measurements of almost 1000 raisins. The idea is that we can use these measurements to predict the type of raisin not in this dataset if we ONLY know the measurements.

## Exploratory Data Analysis:

Before we do any prediction, it's a good idea to get an idea of the data. 

We can easily plot a histogram of any of the columns of our data. The different colors represent different types of raisins. 

"""
columns = list(df.columns)[:-1]
column = st.selectbox("Select a column to explore it's distribution: ", columns)
f = px.histogram(df, x=column, color="Class")
st.plotly_chart(f)

"""
We can see that there are some clear differences between the two kinds of raisins even if we can't exactly describe it. What are some observations you have?

## Prediction

"""


