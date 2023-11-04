import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import nltk
from string import punctuation
import re
from wordcloud import WordCloud
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

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

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2)

# Tokenize the input reviews for both training and validation sets
train_encodings = tokenizer(list(train_df['clean_review']), truncation=True, padding=True)
val_encodings = tokenizer(list(val_df['clean_review']), truncation=True, padding=True)

# Create PyTorch datasets
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, list(train_df['Review_rating']))
val_dataset = ReviewDataset(val_encodings, list(val_df['Review_rating']))

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Set up the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=1e-5)
epochs = 10

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {key: val for key, val in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {key: val for key, val in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            val_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

# Save the trained model
model.save_pretrained('Models/bert_sentiment_analysis')

# Once the model is trained, you can use it for predictions and inference on new data.