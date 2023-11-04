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
# Sentiment analysis using BERT model
sentiment_model = pipeline("sentiment-analysis")


def analyze_sentiment(text):
    sentiment = sentiment_model(text)[0]
    emotion = sentiment['label']
    score = sentiment['score']
    return emotion, score

# Apply sentiment analysis to 'clean_review' column

# if os.path.isfile('Src/shoe_cleanData_semantic.csv'):
#     df = pd.read_csv("Src/shoe_cleanData_semantic.csv")
# else:
#     df[['Emotion', 'Sentiment_Score']] = df['clean_review'].apply(analyze_sentiment).apply(pd.Series)
#     df.to_csv('Src/shoe_cleanData_semantic.csv', index=False)

df[['Emotion', 'Sentiment_Score']] = df['clean_review'].apply(analyze_sentiment).apply(pd.Series)
df.to_csv('Src/shoe_cleanData_semantic.csv', index=False)

# ================================================================
# 4. Recommendations based on sentiment analysis
# Group the DataFrame by product_id
grouped_df = df.groupby('Product_id')

recommendations = []  # List to store recommendations

# Iterate over each product group
for product_id, group in grouped_df:
    # Extract sentiments and emotions
    sentiments = group['Sentiment_Score']
    emotions = group['Emotion']
    reviews = group['clean_review']

    # Calculate overall sentiment and emotion distribution
    overall_sentiment = np.mean(sentiments)
    emotion_distribution = emotions.value_counts(normalize=True)

    # Create a recommendation string for the product
    recommendation = f"Product ID: {product_id}\n"
    recommendation += "==============\n"
    recommendation += f"Overall Sentiment: {overall_sentiment}\n"
    recommendation += "Emotion Distribution:\n"
    recommendation += f"{emotion_distribution}\n"
    recommendation += "--------------\n"

    # Recommendations based on sentiment and emotion analysis
    if overall_sentiment > 0.5:
        recommendation += "Recommendations:\n"
        recommendation += "- Capitalize on positive reviews by featuring them prominently.\n"
        recommendation += "- Use positive testimonials in marketing campaigns.\n"
        # Add more recommendations based on the specific product and sentiment

        # Concrete recommendations based on the reviews
        positive_reviews = reviews[sentiments > 0.5]
        if not positive_reviews.empty:
            recommendation += "Positive Reviews:\n"
            recommendation += f"{positive_reviews.to_string(index=False)}\n"
            # Add more specific recommendations based on the positive reviews

    elif overall_sentiment < 0.5:
        recommendation += "Recommendations:\n"
        recommendation += "- Address concerns raised in negative reviews promptly.\n"
        recommendation += "- Use negative feedback as an opportunity to improve.\n"
        # Add more recommendations based on the specific product and sentiment

        # Concrete recommendations based on the reviews
        negative_reviews = reviews[sentiments < 0.5]
        if not negative_reviews.empty:
            recommendation += "Negative Reviews:\n"
            recommendation += f"{negative_reviews.to_string(index=False)}\n"
            # Add more specific recommendations based on the negative reviews

    else:
        recommendation += "Recommendations:\n"
        recommendation += "- Engage with customers who provided neutral feedback for further insights.\n"
        # Add more recommendations based on the specific product and sentiment

    recommendation += "==============\n\n"

    recommendations.append(recommendation)

# Save the recommendations to a CSV file
recommendations_df = pd.DataFrame(recommendations, columns=["Recommendations"])
recommendations_df.to_csv(
    'Src/shoe_cleanData_recommendations.csv', index=False)

print(df.head())
