#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/master/C3/W1/ungraded_labs/C3_W1_Lab_3_sarcasm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Ungraded Lab: Tokenizing the Sarcasm Dataset
#
# In this lab, you will be applying what you've learned in the past two exercises to preprocess the [News Headlines Dataset for Sarcasm Detection](https://kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home). This contains news headlines which are labeled as sarcastic or not. You will revisit this dataset in later labs so it is good to be acquainted with it now.

# ## Download and inspect the dataset
#
# First, you will fetch the dataset and preview some of its elements.

# In[ ]:


# Download the dataset
get_ipython().system('wget https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json')


# The dataset is saved as a [JSON](https://json.org/json-en.html) file and you can use Python's [`json`](https://docs.python.org/3/library/json.html) module to load it into your workspace. The cell below unpacks the JSON file into a list.

# In[ ]:


import json

# Load the JSON file
with open("./sarcasm.json", 'r') as f:
    datastore = json.load(f)


# You can inspect a few of the elements in the list. You will notice that each element consists of a dictionary with a URL link, the actual headline, and a label named `is_sarcastic`. Printed below are two elements with contrasting labels.

# In[ ]:


# Non-sarcastic headline
print(datastore[0])

# Sarcastic headline
print(datastore[20000])


# With that, you can collect all urls, headlines, and labels for easier processing when using the tokenizer. For this lab, you will only need the headlines but we included the code to collect the URLs and labels as well.

# In[ ]:


# Initialize lists
sentences = []
labels = []
urls = []

# Append elements in the dictionaries into each list
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])


# ## Preprocessing the headlines
#
# You can convert the `sentences` list above into padded sequences by using the same methods you've been using in the past exercises. The cell below generates the `word_index` dictionary and generates the list of padded sequences for each of the 26,709 headlines.

# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the Tokenizer class
tokenizer = Tokenizer(oov_token="<OOV>")

# Generate the word index dictionary
tokenizer.fit_on_texts(sentences)

# Print the length of the word index
word_index = tokenizer.word_index
print(f'number of words in word_index: {len(word_index)}')

# Print the word index
print(f'word_index: {word_index}')
print()

# Generate and pad the sequences
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')

# Print a sample headline
index = 2
print(f'sample headline: {sentences[index]}')
print(f'padded sequence: {padded[index]}')
print()

# Print dimensions of padded sequences
print(f'shape of padded sequences: {padded.shape}')


# This concludes the short demo on using text data preprocessing APIs on a relatively large dataset. Next week, you will start building models that can be trained on these output sequences. See you there!
