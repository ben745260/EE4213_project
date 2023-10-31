import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

file = pd.read_csv("Src/Shoes_Data.csv")

df = file[["reviews", "reviews_rating"]]

product_id =[]
reviews  = []
rates = []

for j in df.index:
    lst = [i for i in df.iloc[j].reviews.split('||')]
    lst2 = [i for i in df.iloc[j].reviews_rating.split('||')]
    for k in lst:
        product_id.append(j+1)
        reviews.append(k)
    for l in lst2:
        rates.append(l)
        
df = pd.DataFrame(list(zip(product_id, reviews, rates)),
               columns =["Product_id", 'Review', 'Review_rating'])

print(df)