# Transformer_Autoencoder
Implementation of Transformer with full Autoencoder

# Requirements
torch == 2.0.1

numpy = 1.23.5

transformers = 4.35.2

# Collab installations
!pip install transformers datasets


# Description
The Transformer Autoencoder is the full architecture of the Transformer model, this model takes into account the encoder only and decoder only parts of the transformer and connects them together to build the regular autoencoder, this model will generally be use for sequence to sequence task such as translation. The model begins by embedding the word tokens of a text then uses the positional encoding to keep sequential consistency, then it goes into a multi-head attention blocks which does a couple of matrix multiplication as well as take advantage of an identity layer, proceeds to go through a feedforward network that will then be use for feature extraction so that it can be use for the decoder model. In the decoder model we do the same procedure as the encoder such as embedding the word tokens, use the positional encoding for maintaining the sequence of words in a text, then we take advantage of causal multi-head-attention blocks which ensures that the transformer is autoregressive therefor becoming a time-series model. Additionally, the model proceeds to a regular multi-head-attention block with the features that come from the causal multi-head-attention block and are combined with the features of the encoder model. We finally proceed to predict the next token words based on the features of both the decoder and encoder model making the transformer model a powerfull architecture.

# Dataset
Harry Potter Books

# Tokenizer
Distilbert

# Architecture
Regular Transformer Model

# optimizer
Adam

# loss function
Cross Entropy Loss

# Text Result:
