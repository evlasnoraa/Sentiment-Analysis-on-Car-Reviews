# Sentiment Analysis on Car Reviews

**Original Project - January 2022**

This project was originally developed as part of an assessment during my Master’s program at the University of Bath. The overarching goal was to perform sentiment analysis on car reviews data using Natural Language Processing (NLP) techniques, including methods like Bag of Words, stemming, and tokenization.

## Project Dataset
For this project, we were provided with a large dataset of reviews for Ford motor vehicles, stored in `car_reviews.csv`. Each review is labeled as either `Pos` (positive) or `Neg` (negative), indicating the sentiment expressed. We assume that these labels are reliable indicators of sentiment, with no neutral reviews in the dataset. The file contains 1,382 reviews in total, split evenly between positive and negative sentiments (691 each).

## Assessment Tasks
The project assessment was divided into two main tasks:

### Task 1
For Task 1, we were instructed to complete the following goals:

1. Avoid issues of data leakage.
2. Identify and remove stop words (i.e., exclude any punctuations or words unlikely to impact sentiment).
3. Ensure that the remaining words are case-insensitive.
4. Use the Bag of Words technique, creating a vector for each review as input for the classifier. This vector could either:
   - Use binary values to indicate word/stem occurrence, or
   - Provide a numerical count of the occurrences for each word/stem.
5. Implement stemming to treat minor word variations as one word in the Bag of Words.
6. Use a Naïve Bayes classifier, with 80% of the reviews (1,106) as training data.

### Task 2
For Task 2, we were given the following objectives:

1. Identify an alternative classification algorithm OR apply modifications to the Naïve Bayes implementation. For example, experimenting with `n-grams` (multi-word phrases) of different sizes.
2. Implement this improvement and compare the results with the initial Naïve Bayes classifier.

---

## Project Approach

### Task 1: Naïve Bayes Classifier

I divided Task 1 into the following steps:

#### Step 1: Splitting the Data (Achieved Goals 1 and 6)
- **Goal**: Split the data while avoiding data leakage.
- **Process**: I randomized the dataset by shuffling it and then split it into training and test sets as specified.

#### Step 2: Data Cleanup (Achieved Goals 2, 3, and 5)
- **Goal**: Clean the data by removing stop words, handling punctuation, and making words case-insensitive.
- **Process**: Leveraged `nltk` functions such as `nltk.corpus.stopwords`, `PorterStemmer`, and `nltk.word_tokenize` to preprocess the text data. This approach significantly reduced data processing time.

#### Step 3: Creating Vectors for the Algorithm (Achieved Goal 4)
- **Goal**: Create input vectors for the Bag of Words model.
- **Process**: Created the Bag of Words model as per the specifications above, counting the occurrences of each word/stem in the reviews.

#### Step 4: Implementing the Naïve Bayes Classifier (Achieved Goal 6)
- **Goal**: Implement the Naïve Bayes classification algorithm.
- **Process**: Used the `MultinomialNB` function from `scikit-learn` to build and train the Naïve Bayes model. To evaluate its performance, I used `confusion_matrix` from `scikit-learn`.

### Task 2: Logistic Regression

In Task 2, I reused the data prepared in the first three steps of Task 1 and implemented **Logistic Regression** as an alternate classifier using `scikit-learn`’s `LogisticRegression` function.

---

## Results and Analysis

### Confusion Matrices (Percent-Wise)
Below are the percent-wise confusion matrices for both classifiers:

#### Naïve Bayes
|               | Predicted Neg | Predicted Pos |
|---------------|---------------|---------------|
| **Actual Neg**| 38.41%        | 10.14%        |
| **Actual Pos**| 8.70%         | 42.75%        |

#### Logistic Regression
|               | Predicted Neg | Predicted Pos |
|---------------|---------------|---------------|
| **Actual Neg**| 37.68%        | 10.14%        |
| **Actual Pos**| 9.42%         | 42.75%        |

### Summary of Classifier Accuracy
- **Naïve Bayes Accuracy**: 81.16%
- **Logistic Regression Accuracy**: 80.43%

### Interpretation
The Naïve Bayes classifier achieved a slightly higher accuracy (81.16%) than the Logistic Regression classifier (80.43%). Both models performed similarly in correctly predicting positive and negative sentiments, though Naïve Bayes exhibited a marginally better performance in this sentiment classification task.

---

## Tools and Libraries
This project utilized the following libraries:
- `nltk` for natural language processing (e.g., `stopwords`, `PorterStemmer`, `word_tokenize`).
- `scikit-learn` for machine learning algorithms, specifically `MultinomialNB` for Naïve Bayes and `LogisticRegression` for logistic regression.

## Conclusion
This project provided a solid foundation in sentiment analysis and natural language processing using Python. Implementing both Naïve Bayes and Logistic Regression allowed for a comparative analysis, highlighting strengths and areas of improvement in each classifier’s performance on sentiment-labeled car reviews.
