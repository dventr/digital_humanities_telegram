# digital_humanities_telegram
Scripts for building a Telegram Corpus and analyzing the data with WordEmbeddings and BERTopic

The data is not available on this Github but can be requested.

The following selection of scripts provides tools for analyzing and extracting insights from Telegram data.

The scripts are based on other projects. Before using the scripts, one has to follow the installation protocol from the following pages.
- Telegram download https://github.com/amiryousefi/telegram-analysis
- BERTopic modeling https://maartengr.github.io/BERTopic/index.html
- word embedding analysis https://gitlab.uzh.ch/noah.bubenhofer/kodup-germanistik/-/tree/master/4._Korpusanalyse/Word_Embeddings

With this toolkit, users can gain valuable insights into various aspects of Telegram messages and their concept, enabling researchers to understand the specialized vocabulary present in Telegram data.

# Overview of Scripts
## Telegram
Different scripts can be used to analyze the data of Telegram channels
- json_download.py can be used to gather data from Telegram
- subcorpus.py is used to reduce data and create a subcorpus
- network_gephi.py creates files for network analysis in Gephi
- id_language.py checks the language of each channel
- creation_telegram.py gives the creation date of each channel

## WordEmbeddings and Telegram
Input files needed in .model format. Instructions can be found on https://gitlab.uzh.ch/noah.bubenhofer/kodup-germanistik/-/tree/master/4._Korpusanalyse/Word_Embeddings
- next_neighbors.py
- cosine_similarity.py

## BERTopic and Telegram
Input files can be generated with json_files from Telegram (json_download.py) and then a subcorpus needs to be created with subcorpus.py (.pkl file as output)
This input file can be used to run topic_bert.py
