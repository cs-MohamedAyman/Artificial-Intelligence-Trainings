#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/master/C3/W1/ungraded_labs/C3_W1_Lab_2_sequences_basic.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Ungraded Lab: Generating Sequences and Padding
#
# In this lab, you will look at converting your input sentences into a sequence of tokens. Similar to images in the previous course, you need to prepare text data with uniform size before feeding it to your model. You will see how to do these in the next sections.

# ## Text to Sequences
#
# In the previous lab, you saw how to generate a `word_index` dictionary to generate tokens for each word in your corpus. You can use then use the result to convert each of the input sentences into a sequence of tokens. That is done using the [`texts_to_sequences()`](https://tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer#texts_to_sequences) method as shown below.

# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define your input texts
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

# Initialize the Tokenizer class
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")

# Tokenize the input sentences
tokenizer.fit_on_texts(sentences)

# Get the word index dictionary
word_index = tokenizer.word_index

# Generate list of token sequences
sequences = tokenizer.texts_to_sequences(sentences)

# Print the result
print("\nWord Index = " , word_index)
print("\nSequences = " , sequences)


# ## Padding
#
# As mentioned in the lecture, you will usually need to pad the sequences into a uniform length because that is what your model expects. You can use the [pad_sequences](https://tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences) for that. By default, it will pad according to the length of the longest sequence. You can override this with the `maxlen` argument to define a specific length. Feel free to play with the [other arguments](https://tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences#args) shown in class and compare the result.

# In[ ]:


# Pad the sequences to a uniform length
padded = pad_sequences(sequences, maxlen=5)

# Print the result
print("\nPadded Sequences:")
print(padded)


# ## Out-of-vocabulary tokens
#
# Notice that you defined an `oov_token` when the `Tokenizer` was initialized earlier. This will be used when you have input words that are not found in the `word_index` dictionary. For example, you may decide to collect more text after your initial training and decide to not re-generate the `word_index`. You will see this in action in the cell below. Notice that the token `1` is inserted for words that are not found in the dictionary.

# In[ ]:


# Try with words that the tokenizer wasn't fit to
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

# Generate the sequences
test_seq = tokenizer.texts_to_sequences(test_data)

# Print the word index dictionary
print("\nWord Index = " , word_index)

# Print the sequences with OOV
print("\nTest Sequence = ", test_seq)

# Print the padded result
padded = pad_sequences(test_seq, maxlen=10)
print("\nPadded Test Sequence: ")
print(padded)


# This concludes another introduction to text data preprocessing. So far, you've just been using dummy data. In the next exercise, you will be applying the same concepts to a real-world and much larger dataset.
