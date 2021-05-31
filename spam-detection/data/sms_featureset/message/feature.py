from typing import Any
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from layer import Dataset


def build_feature(sdf: Dataset("spam_messages")) -> Any:
    init()

    df = sdf.to_pandas()
    feature_data = df[["id", "message"]]
    feature_data = feature_data.assign(clean_message=feature_data["message"].apply(text_cleaning))
    feature_data.drop(columns=["message"], inplace=True)

    return feature_data


def init():
    nltk.download("stopwords")
    nltk.download("wordnet")


def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    stop_words = stopwords.words('english')

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r"ur", " your ", text)
    text = re.sub(r" nd ", " and ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" tkts ", " tickets ", text)
    text = re.sub(r" c ", " can ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r'http\S+', ' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)  # remove numbers
    text = re.sub(r" u ", " you ", text)
    text = text.lower()  # set in lowercase

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    return text
