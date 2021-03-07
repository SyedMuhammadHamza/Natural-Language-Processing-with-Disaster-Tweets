# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 21:55:29 2021

@author: Hamza
"""


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from nltk.corpus import stopwords
from collections import defaultdict
import nltk
nltk.download('stopwords')
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.initializers import Constant
from keras.optimizers import Adam



# Count unique words
def counter_word(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count
def plot_LSA(test_data, test_labels, plot=True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ["orange", "blue", "blue"]
    if plot:
        plt.scatter(
            lsa_scores[:, 0],
            lsa_scores[:, 1],
            s=8,
            alpha=0.8,
            c=test_labels,
            cmap=matplotlib.colors.ListedColormap(colors),
        )
        red_patch = mpatches.Patch(color="orange", label="Irrelevant")
        green_patch = mpatches.Patch(color="blue", label="Disaster")
        plt.legend(handles=[red_patch, green_patch], prop={"size": 16})

stemmer = PorterStemmer()
def stemming(text):
    text = [stemmer.stem(word) for word in text.split()]

    return " ".join(text)

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)

def remove_html(text):
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)

def remove_emoji(string):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)

def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)
def remove_stopwords(text):
    from nltk.corpus import stopwords
    stop = set(stopwords.words("english"))
    text = [word.lower() for word in text.split() if word.lower() not in stop]

    return " ".join(text)



def get_top_text_ngrams(corpus, ngrams=(1, 1), nr=None):
    """
    Creates a bag of ngrams and counts ngram frequency.
    
    Returns a sorted list of tuples: (ngram, count)
    """
    vec = CountVectorizer(ngram_range=ngrams).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:nr]


def create_corpus(df, target):
    """
    Create corpus based on the target.
    """
    corpus = []

    for x in df[df["target"] == target].text.str.split():
        for i in x:
            corpus.append(i)
    return corpus


def generate_ngrams(text, n_gram=1, stop=True):
    """
    Simple n-gram generator.
    """
    stop = set(stopwords.words("english")) if stop else {} #if stop is true then set stop ptherwise ignore it

    token = [
        token for token in text.lower().split(" ") if token != "" if token not in stop
    ]
    z = zip(*[token[i:] for i in range(n_gram)])
    ngrams = [" ".join(ngram) for ngram in z]

    return ngrams
def plot_target_based_features(feature):
    x1 = train[train.target == 1][feature]
    x2 = train[train.target == 0][feature]
    plt.figure(1, figsize=(16, 8))
    plt.subplot(1, 1, 1)
    _ = plt.hist(x2, alpha=0.5, color="grey", bins=50)
    txt1 = "target variable categories distribution for feature {fe}".format(fe=feature)
    plt.title(label=txt1, 
          loc="center",
          fontsize=40, 
          color="green") 
    _ = plt.hist(x1, alpha=0.7, color="red", bins=50)

    return _





train = pd.read_csv(f"data/train.csv")
test = pd.read_csv(f"data/test.csv")
train.head().T


#                                        Exploratory data analysis

# Handling missing values
# When inplace = True , the data is modified in place, which means it will return nothing and the dataframe is now updated. When inplace = False , which is the default, then the operation is performed and it returns a copy of the object. You then need to save it to something

train.fillna(1, inplace=True)
test.fillna(1, inplace=True)
train.keyword.value_counts()[:5]

# distribution of the target variable
disasters = train[train.target == 1].shape[0] #number of post with target as disaster shape[0] number of row 
non_disasters = train[train.target == 0].shape[0]#number of post with target as non-disaster
# train.Id is a pandas Series and is one dimensional. train is a pandas DataFrame and is two dimensional. shape is an attribute that both DataFrames and Series have. It is always a tuple. For a Series the tuple has only only value (x,). For a DataFrame shape is a tuple with two values (x, y). So train.Id.shape[0] would also return 1467. However, train.Id.shape[1] would produce an error while train.shape[1] would give you the number of columns in train.
# Furthermore, pandas Panel objects are three dimensional and shape for it returns a tuple (x, y, z)

print(disasters)

plt.figure(1, figsize=(16, 8))
plt.subplot(1, 2, 1)#((nrows, ncols, fig index))
_ = plt.bar(["Disasters", "Non-disasters"], [disasters, non_disasters])

train.info()
# creating anoter colunm in our trianset named text_len for checking the number of characters in the text feature
train["text_len"] = train.text.map(lambda x: len(x))
train.info()
_ = plot_target_based_features("text_len")
train["words_count"] = train.text.str.split().map(lambda x: len(x))
_ = plot_target_based_features("words_count")


# Count the number of unique words
train["unique_word_count"] = train.text.map(lambda x: len(set(str(x).split())))
_ = plot_target_based_features("unique_word_count")

# Punctuation count
import string

train["punctuation_count"] = train["text"].map(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)

_ = plot_target_based_features("punctuation_count")

# looking at these Plots one thing that we can tell is our target variable categories have very different distributions that are good news from Ml Models perspective,
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


#                                                         N-gram analysis


# In our case when a new word is encountered and is missing from the mapping, the default_factory function calls int() to supply a default count of zero and then, the increment operation builds up the count.
disaster_unigrams = defaultdict(int)
nondisaster_unigrams = defaultdict(int)
for text in train[train.target == 1].text:
    for word in generate_ngrams(text):
        disaster_unigrams[word] += 1

for text in train[train.target == 0].text:
    for word in generate_ngrams(text):
        nondisaster_unigrams[word] += 1

print(disaster_unigrams)
print(disaster_unigrams.items())

df_disaster_unigrams = pd.DataFrame(
    sorted(disaster_unigrams.items(), key=lambda x: x[1], reverse=True)
)
df_nondisaster_unigrams = pd.DataFrame(
    sorted(nondisaster_unigrams.items(), key=lambda x: x[1], reverse=True)
)


d1 = df_disaster_unigrams[0][:10]
d2 = df_disaster_unigrams[1][:10]

nd1 = df_nondisaster_unigrams[0][:10]
nd2 = df_nondisaster_unigrams[1][:10]


plt.figure(1, figsize=(16, 4))
plt.subplot(2, 1, 1)
_ = plt.bar(d1, d2)

plt.title(label="Unigram", 
          loc="center",
          fontsize=40, 
          color="green") 

plt.subplot(2, 1, 2)
_ = plt.bar(nd1, nd2)

disaster_bigrams = defaultdict(int)
nondisaster_bigrams = defaultdict(int)

for text in train[train.target == 1].text:
    for word in generate_ngrams(text, n_gram=2):
        disaster_bigrams[word] += 1

for text in train[train.target == 0].text:
    for word in generate_ngrams(text, n_gram=2):
        nondisaster_bigrams[word] += 1

df_disaster_bigrams = pd.DataFrame(
    sorted(disaster_bigrams.items(), key=lambda x: x[1])[::-1]
)
df_nondisaster_bigrams = pd.DataFrame(
    sorted(nondisaster_bigrams.items(), key=lambda x: x[1])[::-1]
)



d1 = df_disaster_bigrams[0][:10]
d2 = df_disaster_bigrams[1][:10]

nd1 = df_nondisaster_bigrams[0][:10]
nd2 = df_nondisaster_bigrams[1][:10]

plt.figure(1, figsize=(16, 4))
plt.subplot(2, 1, 1)
_ = plt.bar(d1, d2)

plt.title(label="Bigram", 
          loc="center",
          fontsize=40, 
          color="green") 

plt.subplot(2, 1, 2)
_ = plt.bar(nd1, nd2)




disaster_bigrams = defaultdict(int)
nondisaster_bigrams = defaultdict(int)

for text in train[train.target == 1].text:
    for word in generate_ngrams(text, n_gram=3):
        disaster_bigrams[word] += 1

for text in train[train.target == 0].text:
    for word in generate_ngrams(text, n_gram=3):
        nondisaster_bigrams[word] += 1

df_disaster_bigrams = pd.DataFrame(
    sorted(disaster_bigrams.items(), key=lambda x: x[1])[::-1]
)
df_nondisaster_bigrams = pd.DataFrame(
    sorted(nondisaster_bigrams.items(), key=lambda x: x[1])[::-1]
)



d1 = df_disaster_bigrams[0][:10]
d2 = df_disaster_bigrams[1][:10]

nd1 = df_nondisaster_bigrams[0][:10]
nd2 = df_nondisaster_bigrams[1][:10]

plt.figure(1, figsize=(16, 4))
plt.subplot(2, 1, 1)
_ = plt.bar(d1, d2)

plt.title(label="Trigram", 
          loc="center",
          fontsize=40, 
          color="green") 

plt.subplot(2, 1, 2)
_ = plt.bar(nd1, nd2)


# we've divided our corpus-based into target variable categories and got the desired result as most of the N-gram of one category was missing from the other now let's check the most common bigrams in a whole corpus.





top_text_bigrams = get_top_text_ngrams(train.text, ngrams=(2, 2), nr=10)


x, y = zip(*top_text_bigrams)
plt.figure(1, figsize=(16, 8))
plt.subplot(1, 1, 1)
plt.title(label="Unigram of whole corpus independent of target variable category", 
          loc="center",
          fontsize=40, 
          color="green") 
plt.bar(x, y)


corpus0 = create_corpus(df=train, target=0)
corpus1 = create_corpus(df=train, target=1)

punc0 = defaultdict(int)
for word in corpus0:
    if word in string.punctuation:
        punc0[word] += 1

punc1 = defaultdict(int)
for word in corpus1:
    if word in string.punctuation:
        punc1[word] += 1

top0 = sorted(punc0.items(), key=lambda x: x[1], reverse=True)[:10]
top1 = sorted(punc1.items(), key=lambda x: x[1], reverse=True)[:10]

x0, y0 = zip(*top0)
x1, y1 = zip(*top1)

plt.figure(1, figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.bar(x0, y0)
plt.subplot(1, 2, 2)
plt.bar(x1, y1)

# we've seen a lot of URLs and HTML elements during our N-gram analysis its time to get ride of them


train["text"] = train.text.map(lambda x: remove_URL(x))
train["text"] = train.text.map(lambda x: remove_html(x))
train["text"] = train.text.map(lambda x: remove_emoji(x))
train["text"] = train.text.map(lambda x: remove_punct(x))
train["text"] = train["text"].map(remove_stopwords)


# WordCloud for each of the target class
corpus0 = create_corpus(df=train, target=0)
corpus1 = create_corpus(df=train, target=1)
word_cloud0 = WordCloud(background_color="white", max_font_size=80).generate(
    " ".join(corpus0[:50])
)
word_cloud1 = WordCloud(background_color="white", max_font_size=80).generate(
    " ".join(corpus1[:50])
)

plt.figure(1, figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(word_cloud1)
plt.title(label="WordCloud for disasters ", 
          loc="center",
          fontsize=40, 
          color="green")
plt.subplot(1, 2, 2)
plt.imshow(word_cloud0)
plt.title(label="WordCloud for non disasters ", 
          loc="center",
          fontsize=40, 
          color="green")
           
# Performing Stemming

train["text"] = train["text"].map(stemming)
corpus0 = create_corpus(df=train, target=0)
corpus1 = create_corpus(df=train, target=1)

word_cloud0 = WordCloud(background_color="white", max_font_size=80).generate(
    " ".join(corpus0[:50])
)
word_cloud1 = WordCloud(background_color="white", max_font_size=80).generate(
    " ".join(corpus1[:50])
)

plt.figure(1, figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(word_cloud1)
plt.title(label="WordCloud after stemming for disasters ", 
          loc="center",
          fontsize=20, 
          color="green")
plt.subplot(1, 2, 2)
plt.imshow(word_cloud0)
plt.title(label="WordCloud after stemming for non disasters ", 
          loc="center",
          fontsize=20, 
          color="green")
          
# Tokenization
tokenizer = TreebankWordTokenizer()

train["tokens"] = train["text"].map(tokenizer.tokenize)
train[["text", "tokens"]].head(10)


#                                                     Bag of words

# Bag of words embeds each sentence as a list of 0s, with a 1 at each index corresponding to a word present in the sentence.

train_counts, count_vectorizer = count_vect(train["text"])
test_counts = count_vectorizer.transform(test["text"])
print(train.shape)
print(train_counts.todense().shape)

print(train.text.iloc[0])
print(train_counts.todense()[0][0:].sum())




#                             Embedding and LSTM Recurrent Neural Network

# 	BOW model and TF-IDF model are not good for complicated tasks cause they can't understand the context of words but Word Embedding incorporate the context and meaning of words on top of that the dimensionality is reduced  in case of word embeddings 
text = train.text

counter = counter_word(text)

print(len(counter))
print(counter)

num_words = len(counter)
# Max number of words in a sequence
max_length = 20


# Train / test split
train_size = int(train.shape[0] * 0.8)

train_sentences = train.text[:train_size]
train_labels = train.target[:train_size]

test_sentences = train.text[train_size:]
test_labels = train.target[train_size:]


tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index

print(word_index)

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_sequences[0]


train_padded = pad_sequences(
    train_sequences, maxlen=max_length, padding="post", truncating="post"
)

train_padded[0]

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(
    test_sequences, maxlen=max_length, padding="post", truncating="post"
)

print(train.text[0])
print(train_sequences[0])


# Check inverse
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

decode(train_sequences[0])
model = Sequential()

model.add(Embedding(num_words, 32, input_length=max_length))
model.add(LSTM(64, dropout=0.1))
model.add(Dense(1, activation="sigmoid"))

optimizer = Adam(learning_rate=3e-4)

model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

model.summary()

model = model.fit(
    train_padded, train_labels, epochs=20, validation_data=(test_padded, test_labels),
)
