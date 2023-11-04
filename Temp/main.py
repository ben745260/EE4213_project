import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import nltk
from string import punctuation
import re
from wordcloud import WordCloud
from transformers import pipeline
import os
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
import torch

warnings.filterwarnings('ignore')

# Init the dataset to review and rating only
# ================================================================
# 1. Data acquisition
file = pd.read_csv("Src/Shoes_Data.csv")
# ================================================================
# 2. Data processing
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

# Cleaning functions
def lower(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', punctuation))

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

def remove_non_printable(text):
    text = text.encode("ascii", "ignore")
    return text.decode()

def clean_text(text):
    text = lower(text)
    text = remove_punctuation(text)
    text = remove_digits(text)
    text = remove_emoji(text)
    text = remove_non_printable(text)
    return text

# Apply the cleaning function to 'Review' column
df['clean_review'] = df['Review'].apply(clean_text)

# Save the cleaned DataFrame to a CSV file
df.drop("Review", axis=1, inplace=True)
df.to_csv('Src/shoe_cleanData.csv', index=False)

# ================================================================
# 3. Model deployment
# Sentiment analysis using BERT model 
# ================================================================
# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Define a function to perform sentiment analysis using BERT
def analyze_sentiment(review):
    inputs = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).tolist()[0]
    
    sentiment = 'Positive' if probabilities[1] > probabilities[0] else 'Negative'
    return sentiment

# Apply sentiment analysis to the 'clean_review' column
df['Sentiment'] = df['clean_review'].apply(analyze_sentiment)

# Save the sentiment analysis results to a CSV file
df.to_csv('Src/sentiment_analysis_results.csv', index=False)

# ================================================================
# 4. Recommendations based on sentiment analysis
# Group the DataFrame by product_id
grouped_df = df.groupby('Product_id')

recommendations = []

for product_id, group in grouped_df:
    positive_reviews = group[group['Sentiment'] == 'Positive']
    positive_review_count = len(positive_reviews)
    
    negative_reviews = group[group['Sentiment'] == 'Negative']
    negative_review_count = len(negative_reviews)
    
    total_review_count = len(group)
    
    recommendation = {
        'Product_id': product_id,
        'Positive_Review_Count': positive_review_count,
        'Negative_Review_Count': negative_review_count,
        'Total_Review_Count': total_review_count
    }
    
    recommendations.append(recommendation)

recommendations_df = pd.DataFrame(recommendations)

# Save the recommendations to a CSV file
recommendations_df.to_csv('Src/recommendations_results.csv', index=False)