import pandas as pd
import warnings
from string import punctuation
import re
from transformers import pipeline
import os
import matplotlib.pyplot as plt

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

# Save the cleaned and analyzed DataFrame to a CSV file
df.drop("Review", axis=1, inplace=True)
df.to_csv('Src/shoe_cleanData.csv', index=False)


# ================================================================
# 3. Model deployment
# Sentiment analysis using DistiBERT model
# (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    sentiment = sentiment_model(text)[0]
    emotion = sentiment['label']
    score = sentiment['score']
    return emotion, score


# Apply sentiment analysis to 'clean_review' column
if os.path.isfile('Src/DistilBERT/shoe_cleanData_semantic.csv'):
    df = pd.read_csv("Src/DistilBERT/shoe_cleanData_semantic.csv")
else:
    df[['Emotion', 'Sentiment_Score']] = df['clean_review'].apply(
        analyze_sentiment).apply(pd.Series)
    df.to_csv('Src/DistilBERT/shoe_cleanData_semantic.csv', index=False)


# ================================================================
# 4. Recommendations based on sentiment analysis
# Group the DataFrame by product_id
print("Part 4")

# Group the DataFrame by 'Product_id'
grouped_df = df.groupby('Product_id').agg(
    {'Emotion': ['count', lambda x: (x == 'POSITIVE').mean()]})
grouped_df.columns = ['Review_count', 'Average_sentiment']
grouped_df.reset_index(inplace=True)

# Sort the DataFrame based on the 'Review_count' and 'Average_sentiment'
sorted_df = grouped_df.sort_values(
    ['Review_count', 'Average_sentiment'], ascending=[False, False])

# Save the recommendations results to a CSV file
sorted_df.to_csv('Src/DistilBERT/shoe_cleanData_recommendations.csv', index=False)

# Recommendation based on Positive Sentiment
positive_recommendations = df.groupby('Product_id').filter(lambda x: (x['Emotion'] == 'POSITIVE').mean() > 0.8)
positive_recommendations = positive_recommendations.drop_duplicates(subset='Product_id')

# Recommendation based on Negative Sentiment
negative_recommendations = df.groupby('Product_id').filter(lambda x: (x['Emotion'] == 'NEGATIVE').mean() > 0.8)
negative_recommendations = negative_recommendations.drop_duplicates(subset='Product_id')

# Recommendation based on Sentiment Score
average_sentiment = df.groupby('Product_id')['Sentiment_Score'].mean()
sentiment_recommendations = average_sentiment.nlargest(3).index.tolist()

# Save recommendations to a text file
recommendations_path = 'Src/DistilBERT/recommendations.txt'
with open(recommendations_path, 'w') as file:
    file.write("Positive Sentiment Recommendations:\n")
    file.write(str(positive_recommendations) + "\n\n")
    file.write("Negative Sentiment Recommendations:\n")
    file.write(str(negative_recommendations) + "\n\n")

    file.write("\n\n")
    file.write("Sentiment Score Recommendations:\n")
    file.write(str(sentiment_recommendations) + "\n")