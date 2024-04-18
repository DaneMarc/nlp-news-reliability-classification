# Automated Detection of Misinformation in News Articles

Code for CS4248 Group Project (AY22/23 Sem 2) [[`code`](https://github.com/DaneMarc/nlp-news-reliability-classification)]

By Dane Marc Yap Bagaoisan, Hu Yue, Joshua Emmanuel Teo Rui Zhong, Ni Shenghan, Sajal Vaishnav, Wang Huan

## Abstract

Recent advancements in generative artificial intelligence (AI) capabilities have made content generation easier than ever, exacerbating the spread of misinformation in media. As a possible solution to this issue, our research aims to develop a machine learning classifier that could help by automatically identifying news articles as reliable, satirical, hoax, or propaganda. We theorised that these categories appealed to the reader's emotions differently and thus experimented with different ways of preprocessing text, creating embeddings and enhancing those embeddings with sentiment features. We compared a number of machine learning models and found that a combination of TextCNN and LSTM models performed the best.

## Folder Structure

```
nlp
|-embedding/
|-models/
|-preprocessing/
```

- `embedding/` contains code we used to produce the document embeddings 
- `model/` contains code we used to establish baselines, train, run inference, and test our models  
- `preprocessing/` contains the python notebooks we used to create 13 preprocessing strategies

## Running Instructions
1. Download the folders from [this drive link](https://drive.google.com/drive/folders/1GowTReLmoK986WR57DZgLZlLADET-l3l)
    - These folders contain the various embeddings and pre-processed datasets we produced to train our models
2. Each individual python file or notebook inside the repo is runnable on it's own, given the file paths are setup correctly


## References
1. Compare to The Knowledge: Graph Neural Fake News Detection with External Knowledge
https://aclanthology.org/2021.acl-long.62/
github:https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection

2. Truth of Varying Shades: Analyzing Language in Fake News
and Political Fact-Checking
https://aclanthology.org/D17-1317.pdf

3. Do Sentence Interactions Matter ? Leveraging Sentence Level Representations for Fake News Classification
https://arxiv.org/pdf/1910.12203.pdf
github:https://github.com/MysteryVaibhav/fake_news_semantics
