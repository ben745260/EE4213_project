import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import nltk
from string import punctuation
import re
from wordcloud import WordCloud
warnings.filterwarnings('ignore')

# ================================================================
# Init the dataset to review and rating only


file = pd.read_csv("Src/Shoes_Data.csv")

df = file[["reviews", "reviews_rating"]]

product_id = []
reviews = []
rates = []

for j in df.index:
    lst = [i for i in df.iloc[j].reviews.split('||')]
    lst2 = [i for i in df.iloc[j].reviews_rating.split('||')]
    for k in lst:
        product_id.append(j + 1)
        reviews.append(k)
    for l in lst2:
        rates.append(l)

df = pd.DataFrame(list(zip(product_id, reviews, rates)),
                  columns=["Product_id", 'Review', 'Review_rating'])

# print(df)

# ================================================================
# Getting all unique symbols in text
all_text = str()
for sentence in df['Review'].values:
    all_text += sentence

''.join(set(all_text))


def lower(text):
    return text.lower()


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', punctuation))


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop])

# Removing all words with digits and standalone digits


def remove_digits(text):
    return re.sub(r'\d+', '', text)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Removing all non-printable symbols like "ड", "ட"


def remove_non_printable(text):
    text = text.encode("ascii", "ignore")
    return text.decode()

# One function to clean it all


def clean_text(text):
    text = lower(text)
    text = remove_punctuation(text)
    # text = remove_stopwords(text)
    text = remove_digits(text)
    text = remove_emoji(text)
    text = remove_non_printable(text)
    return text

# Returns first digit entry in a string
def get_first_digit(text):
    match = re.search(r'\d', text)
    return match[0]

df['Review_rating']=df['Review_rating'].apply(get_first_digit)
df['clean_review']=df['Review'].apply(clean_text)
# print(df.head())

all_text_clean = str()
for sentence in df['clean_review'].values:
    all_text_clean += sentence
''.join(set(all_text_clean))


print(df["clean_review"].value_counts())



# ================================================================
# # Plot Key words of every rating to word cloud

# # plt.figure(figsize=(40,25))

# # subset1 = df[df['Review_rating']=='1']
# # text = subset1.clean_review.values
# # cloud1=WordCloud(background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))

# # plt.subplot(3, 2, 1)
# # plt.axis('off')
# # plt.title("1",fontsize=40)
# # plt.imshow(cloud1)

# # subset2 = df[df['Review_rating']=='2']
# # text = subset2.clean_review.values
# # cloud2=WordCloud(background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))

# # plt.subplot(3, 2, 2)
# # plt.axis('off')
# # plt.title("2",fontsize=40)
# # plt.imshow(cloud2)

# # subset3 = df[df['Review_rating']=='3']
# # text = subset3.clean_review.values
# # cloud3=WordCloud(background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))

# # plt.subplot(3, 2, 3)
# # plt.axis('off')
# # plt.title("3",fontsize=40)
# # plt.imshow(cloud3)

# # subset4 = df[df['Review_rating']=='4']
# # text = subset4.clean_review.values
# # cloud4=WordCloud(background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))

# # plt.subplot(3, 2, 4)
# # plt.axis('off')
# # plt.title("4",fontsize=40)
# # plt.imshow(cloud4)

# # subset5 = df[df['Review_rating']=='5']
# # text = subset5.clean_review.values
# # cloud5=WordCloud(background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))

# # plt.subplot(3, 2, 5)
# # plt.axis('off')
# # plt.title("5",fontsize=40)
# # plt.imshow(cloud5)