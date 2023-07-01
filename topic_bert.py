# Further infomrations can be found on the Github page of Maarten Grootendorst
# https://maartengr.github.io/BERTopic/index.html 

# input are .pkl files containing dates and messages from telegram
# output are different .html visualisation, that bertopic can produce

import pickle
import plotly.express as px
import plotly.io as pio
import matplotlib as plt

from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from bertopic.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer

# word at the center of the analysis
wort = 'freiheit'
# Open the file in list format for reading
with open('messages_list_' + wort + '.pkl', 'rb') as f:
    # Use pickle to deserialize the list and load it from the file
    messages_list = pickle.load(f)

# Open the file in list format for reading
with open('date_list_' + wort + '.pkl', 'rb') as f:
    # Use pickle to deserialize the list and load it from the file
    date_list = pickle.load(f)


# create custom embeddings, if there is specific vocabulary present in the data
class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def embed(self, messages_list, verbose=False):
        embeddings = self.embedding_model.encode(messages_list, show_progress_bar=verbose)
        return embeddings 

# Create custom backend
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
custom_embedder = CustomEmbedder(embedding_model=embedding_model)
# this is needed for the visualization of the documents
sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = sentence_model.encode(messages_list, show_progress_bar=False)
# increasing diversity means a more diverse range of words for topic modelling
representation_model = MaximalMarginalRelevance(diversity=0.5)
# default: umap_model = UMAP(n_neighbors=15, n_components=10, metric='cosine', low_memory=False)
# nextneighbors: higher = more global, dimensionality of embedding (high or low no good cluster)
umap_model = UMAP(n_neighbors=20, n_components=12, min_dist=0.0, metric='cosine')
# min cluster set higher than default for bigger clusters (def = 10)
hdbscan_model = HDBSCAN(min_cluster_size=20, metric='euclidean', 
                        cluster_selection_method='eom', prediction_data=True, min_samples=5)
# custom vectorizer, can later be changed to specific options. 
# min_df = how many times a word must occur
# ngrams to capture frequent words that appear together
vectorizer_model = CountVectorizer(min_df=3, ngram_range=(1, 3))
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# upper limit of 100 to limit the topics
topic_model = BERTopic(embedding_model=custom_embedder, representation_model=representation_model, umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model, ctfidf_model=ctfidf_model, nr_topics=100)
# application to messages list
topics, probs = topic_model.fit_transform(messages_list)

topic_model.save("telegram_model_freiheit")
# versions of dependencies and python used. loading and saving model => same dependencies and python
# saved in one version of Bertopic should not be loaded in others
#topic_model = BERTopic.load("telegram_model")



# barchart visualization of top20 topics
fig0 = topic_model.visualize_barchart(top_n_topics=20, n_words=20)
# Convert the figure to HTML
html0 = pio.to_html(fig0)
# Write the HTML to a file
with open('barchart_' + wort + '.html', 'w') as f:
    f.write(html0)

#visualization of the topics and their cluster
#Then, we can call .visualize_topics to create a 2D representation of your topics.
#The resulting graph is a plotly interactive graph which can be converted to HTML:
fig = topic_model.visualize_topics()
# Convert the figure to HTML
html = pio.to_html(fig)
# Write the HTML to a file
with open('topic' + wort + '.html', 'w') as f:
    f.write(html)

# dynamic topic modeling
# generate the topics with the datelist
topics_over_time = topic_model.topics_over_time(messages_list, date_list, nr_bins=20)
# It is advised to keep the number of unique timestamps below 50
# in order to change the datetime format use:  datetime_format="%b%M"
fig1 = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=25)
html1 = pio.to_html(fig1)
# Write the HTML to a file
with open('dynamisch_' + wort + '.html', 'w') as f:
    f.write(html1)


#hierarchical topic modeling
# hierarchical topic modeling
hierarchical_topics = topic_model.hierarchical_topics(messages_list)
fig2 = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
html2 = pio.to_html(fig2)
with open('hierarchisch_' + wort + '.html', 'w') as f:
    f.write(html2)
# merging topics, if the researcher finds similar topics and wants to merge them
#topics_to_merge = [[1, 2],[3, 4]]
#topic_model.merge_topics(docs, topics_to_merge)"""

# Run the visualization with the original embeddings
# each document and the topic visualized
visualize_docs = topic_model.visualize_documents(messages_list, embeddings=embeddings, hide_document_hover=False)
html3 = pio.to_html(visualize_docs)
with open('documents_' + wort + '.html', 'w') as f:
    f.write(html3)
    
#heatmap visualisation of each
heatmap = topic_model.visualize_heatmap()
html4 = pio.to_html(heatmap)
with open('heatmap' + wort + '.html', 'w') as f:
    f.write(html4)

