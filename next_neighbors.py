# based on https://gitlab.uzh.ch/noah.bubenhofer/kodup-germanistik/-/tree/master/4._Korpusanalyse/Word_Embeddings 
# a few modification were made to this script
# input files can be created by following the tutorial from using the following Github page
#  https://gitlab.uzh.ch/noah.bubenhofer/kodup-germanistik/-/tree/master/4._Korpusanalyse/Word_Embeddings 
# output file will be a .txt file with 150 next neighbors of the word "Freiheit" (freedom)
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import re

# word we want to find next neighbors
word = 'Freiheit'
# give the folder a fitting name
output_folder = 'output'
# give the file a fitting name
filename = 'next_neighbors_' + word + '_kd' '.txt'
list_nextn = []

def we_basics(infile):
    model = Word2Vec.load(infile)
    # topn, how many next neighbors we want to have
    sims = model.wv.most_similar(word, topn=150)
    for elem in sims:
        list_nextn.append(elem[0])
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the file in write mode inside the output folder
    # Write the content to the file
    with open(os.path.join(output_folder, filename), "w") as file:
        for item in list_nextn:
            file.write(item + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contains components for simple operations with Word Embedding")
    parser.add_argument("in_file", type=str, help="Input file: Word Embeddings model in the word_vectors format")
    args = parser.parse_args()
    we_basics(args.in_file)
