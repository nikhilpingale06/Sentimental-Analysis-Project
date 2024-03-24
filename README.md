# Sentimental-Analysis-Project
Sentiment Analysis on IMBD dataset using Bidirectional LSTM neural network

1. Introduction:
Sentiment analysis, a subfield of natural language processing, aims to determine the emotional tone behind a piece of text. In this project, sentiment analysis was performed on the IMDB movie review dataset consisting of 50,000 records. The goal was to classify movie reviews as either positive or negative based on their sentiment.

2. Dataset:
The IMDB movie review dataset comprises 50,000 movie reviews, evenly split into 25,000 for training and 25,000 for testing. Each review is labeled as either positive or negative.

3. Methodology:

  *Data Preprocessing:*
  Text data was preprocessed by removing HTML tags, punctuation, and special characters.
  Tokenization was performed to split text into individual words.
  Stop words were removed to reduce noise in the data.

  *Word Embeddings:*
  GloVe (Global Vectors for Word Representation) embeddings were utilized to represent words in a continuous vector space.

  *Model Architecture:*
  Bidirectional LSTM (Long Short-Term Memory) neural network was employed for sentiment classification.
  LSTM units were bidirectional to capture contextual information from both past and future sequences.
  Dropout regularization was applied to prevent overfitting.

  *Training:*
  The TensorFlow library was utilized for building and training the neural network model.
  The training dataset was used to train the model, optimizing it to minimize the loss function.

  *Evaluation:*
  The model's performance was evaluated on the test dataset to assess its accuracy and generalization ability.

4. Results:
The Bidirectional LSTM model achieved an accuracy of **86.75%** on the IMDB movie review dataset, indicating its effectiveness in sentiment analysis.
The model demonstrated robustness in distinguishing between positive and negative sentiments in movie reviews.

5. Libraries Used:
TensorFlow: Used for building and training the neural network model.
NLTK (Natural Language Toolkit): Utilized for text preprocessing tasks such as tokenization and stop word removal.
