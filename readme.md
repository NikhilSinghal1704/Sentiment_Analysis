
# Sentiment Analysis with Bag of Words and VADER

This project performs sentiment analysis on user reviews using two different approaches: Bag of Words and VADER (Valence Aware Dictionary and sEntiment Reasoner).

## Files:

1. **bag_of_words.py**: This script uses the Bag of Words approach to analyze the sentiment of user reviews. It reads user reviews from a CSV file (`user_review.csv`) and predicts their sentiment using a Multinomial Naive Bayes classifier trained on the `test.csv` dataset obtained from [Kaggle](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset).

2. **vader.py**: This script uses the VADER sentiment analysis tool, which is specifically designed for social media text. It directly analyzes the sentiment of user reviews using the VADER sentiment analyzer.

3. **user_review.csv**: This CSV file contains the user reviews to be analyzed. It should have a column named 'review' containing the text of the reviews.

4. **test.csv**: This CSV file is used to train the Bag of Words approach. It contains a dataset of user reviews with sentiment labels obtained from Kaggle.

## Usage:

1. Ensure you have the required dependencies installed. You can install them using pip and the provided `requirements.txt` file:

    ```
    pip install -r requirements.txt
    ```

2. Run either `bag_of_words.py` or `vader.py` depending on the approach you want to use for sentiment analysis.

    ```
    python bag_of_words.py
    ```

    or

    ```
    python vader.py
    ```

3. The script will read the user reviews from `user_review.csv`, perform sentiment analysis, and display the results. Additionally, it will generate a pie chart visualizing the distribution of sentiments among the user reviews.

4. After analysis, both scripts will create a new CSV file named `analyzed_user_reviews.csv` containing the analyzed user reviews along with their predicted sentiments.

5. Additionally, both scripts will generate a pie chart visualizing the distribution of sentiments among the user reviews. The pie chart will be displayed in the console.

## Files:

- **requirements.txt**: This file lists all the dependencies required by the project. You can install them using `pip install -r requirements.txt`.