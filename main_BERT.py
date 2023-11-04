import pandas as pd
import warnings
from string import punctuation
import re
from transformers import pipeline
import os
from transformers import BertTokenizer
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
print("Part 3")

import torch
from transformers import BertConfig, BertModel, BertTokenizer

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
config = BertConfig.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, config=config)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Define a function to perform sentiment analysis on a given text using the loaded model and tokenizer
def perform_sentiment_analysis(text):
    encoding = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_state = outputs.last_hidden_state
    pooled_output = torch.mean(last_hidden_state, dim=1)
    logits = torch.nn.Linear(pooled_output.shape[-1], config.num_labels)(pooled_output)
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment = 'Positive' if probabilities[1] > probabilities[0] else 'Negative'
    return sentiment

if os.path.isfile('Src/BERT/shoe_cleanData_semantic.csv'):
    df = pd.read_csv("Src/BERT/shoe_cleanData_semantic.csv")
else:
    # Apply the sentiment analysis function to the 'clean_review' column in the DataFrame
    df['Sentiment'] = df['clean_review'].apply(perform_sentiment_analysis)
    # Save the sentiment analysis results to a CSV file
    df.to_csv('Src/BERT/shoe_cleanData_semantic.csv', index=False)

# ================================================================
# 4. Recommendations based on sentiment analysis
print("Part 4")

# Group the DataFrame by 'Product_id'
grouped_df = df.groupby('Product_id').agg({'Sentiment': ['count', lambda x: (x == 'Positive').mean()]})
grouped_df.columns = ['Review_count', 'Average_sentiment']
grouped_df.reset_index(inplace=True)

# Sort the DataFrame based on the 'Review_count' and 'Average_sentiment'
sorted_df = grouped_df.sort_values(['Review_count', 'Average_sentiment'], ascending=[False, False])

# Save the recommendations results to a CSV file
sorted_df.to_csv('Src/BERT/shoe_cleanData_recommendations.csv', index=False)