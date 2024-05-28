import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Reading the user review data
reviews_df = pd.read_csv('user_review.csv')

# Extracting reviews from DataFrame
reviews = reviews_df['review']

# Initializing the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

sentiments = []

# Analyzing sentiments for each review
for review in reviews:
    sentiment_scores = analyzer.polarity_scores(review)
    print(sentiment_scores)  # Print the sentiment scores for each review
    del sentiment_scores['compound']  # Remove the compound score
    sentiment = max(sentiment_scores, key=sentiment_scores.get)  # Determine the dominant sentiment
    sentiments.append(sentiment)

# Adding the sentiment column to the DataFrame
reviews_df['sentiment'] = sentiments

# Calculate the distribution of sentiments
sentiment_distribution = reviews_df['sentiment'].value_counts()

# Plot the pie chart for sentiment distribution
plt.figure(figsize=(8, 6))
plt.pie(sentiment_distribution, labels=sentiment_distribution.index, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution')
plt.axis('equal')
plt.show()

# Output
reviews_df.to_csv('analyzed_user_reviews.csv', index=False)
