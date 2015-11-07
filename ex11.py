"""
Name:

Code adapted from Joseph Wilk's semanticpy github, specifically
https://github.com/josephwilk/semanticpy/blob/master/semanticpy/transform/lsa.py

What to try:

1. Look at the plot of novels based on the first two LSA dimensions. 
Are the points evenly spaced or are there outliers?

Yes, some trailing off to the left away from the main clump.

2. Is there a relationship between the first LSA dimension and the length
of the novels?

Yes, the density of the dimension in the novel isn't being considered.

3. Uncomment the line in read_novels() that normalizes the length of novels
and rerun. Does the plot look different?

Yes, it has a curved shape now instead of the clump.

4. By default we're highlighting author gender in the plot. Are there other
values from the metadata file that show interesting patterns?

Yes, in the grad secion we looked at whether using first person narrative would 
produce a pattern, and it seemed to, one stronger than the pattern from gender.

5. Which metadata values are easy to work with, and which are more difficult? Why?

I tried the number of volumes variable and it looks like there is some pattern, but
it is also really hard to track a variable that has more than two or three possible
values when you're just eyeballing it. (Also the plots we're drawing are small...)
Author age and year would have the same issue, trying to do it on the subjects would be
really hard.

"""

import csv
import numpy as np
from scipy import dot
from scipy import linalg
import matplotlib

def read_novels():
    """Read in a file describing metadata and word counts for a number
    of novels, returning the vocabulary, novel metadata, and a matrix of
    word counts for each novel and each word in the vocabulary."""
    novels = []
    
    with open("ota-metadata.tsv") as metadatafile:
        metadata_reader = csv.reader(metadatafile, delimiter='\t')
        
        header_line = metadata_reader.next()
        line_number = 0
        for novel_metadata_line in metadata_reader:
            novel = dict(zip(header_line, novel_metadata_line))
            novel["id"] = line_number
            novels.append(novel)
            line_number += 1
    
    # This reads in every novel's metadata and word counts for the 10000
    # most frequent words
    with open("novel_count_file.tsv") as novelfile:
        novel_reader = csv.reader(novelfile, delimiter='\t') 
        
        # We find the titles for each of the rows, including the word list
        header_line = novel_reader.next()
        vocab = header_line[3:]
        novelmatrix = []
        line_number = 0

        # We read each novel into a dictionary
        for novel_data_line in novel_reader:
            title, author, year = novel_data_line[:3]
            
            ## everything from index 3 on is a string
            ##  representing a word count
            count_strings = novel_data_line[3:]
            word_count_list = np.zeros(len(count_strings))

            for i in range(0, len(count_strings)):
                word_count_list[i] = float(count_strings[ i ])

            ## Stretch each vector so that it has length = 1.0
            #length of the vecotr divided by the length
#uncommented every novel would have lenght one in some dimension
#get a curve shape
# what if divide by the sum of the documents
#coloring gender, doesn't tell you which is which but green probably male
            word_count_list /= np.sqrt(np.sum(word_count_list**2))

            # We'll put the actual words into a matrix, where the
            # ith row is word counts for the ith novel in novels
            novelmatrix.append(word_count_list)
            
            line_number += 1

        # We're also going to convert this to a nifty format
        # for doing math things to it
        novelmatrix = np.transpose(np.array(novelmatrix))
        
        return vocab, novels, novelmatrix

def lsa(document_word_matrix, dimension):
    """
    Take a document-word matrix and retrieve document-concept and concept-word
    matrices from it using latent semantic analysis (LSA).
    """
    # We need to know the shape of our starting document-word matrix in
    # terms of number of rows and columns in order to run LSA.
    rows, cols = document_word_matrix.shape

    #for row in range(rows):
    #    document_word_matrix[row,:] /= math.sqrt()

    # We can't create a matrix bigger than what we started with
    if dimension > rows:
        raise ValueError("Dimension {} too big!".format(dimension))

    # Dimensions also have to be positive
    elif dimension < 1:
        raise ValueError("Dimension {} too small!".format(dimension))

    # We use singular value decomposition to decompose our original
    # document-word matrix into three matrixes that, multiplied together,
    # recreate our original:
    # - word_topic: a matrix with m terms as rows and r "concept"
    #   proportions as columns,
    # - singular_values: a nonnegative diagonal matrix of r rows and r 
    #   columns, and
    # - topic_document: a matrix with r "concepts" as rows and n documents
    #   as columns.
    # Because the singular_values matrix actually only has values on the 
    # diagonal, we just get it as a list of r singular values that would be
    # the diagonal of the matrix in order from greatest to least.
    word_topic, singular_values, topic_document = linalg.svd(document_word_matrix)
    print singular_values

    # Our goal is to reduce the original dimensions of this to the number
    # of concepts or "topics" we want, which we do by discarding all of the
    # columns and rows corresponding to values we don't need. This is
    # straightforward for our word-topic matrix: we throw out all of the
    # columns past the dimension we want.
    lsa_singular_values = singular_values[:dimension]
    lsa_word_topic = word_topic[:,:dimension]
    
    # Our topic-document matrix is a little trickier, because we'd rather
    # have our documents as rows and topics as columns, and right now it's
    # the other way around. So we'll switch it or transpose it.
    lsa_topic_document = topic_document[:dimension,:]
    lsa_document_topic = np.transpose(lsa_topic_document)

    # We can check that we did things right by using our new matrices
    new_singular_matrix = linalg.diagsvd(lsa_singular_values, dimension, dimension)
    transformed_matrix = dot(dot(lsa_word_topic, new_singular_matrix), lsa_topic_document)
    
    # We know that SVD gives us in our singular value matrix the values we care
    # about in order.
    print "Representation error: {}".format(np.sum((document_word_matrix - transformed_matrix)**2))

    return lsa_word_topic, lsa_document_topic

def closest_words(query):
    row_norms = np.sqrt(np.sum(lsa_word_topic**2, axis=1))
    query_index = vocab.index(query)
    query_vector = lsa_word_topic[query_index]
    cosines = np.divide(np.dot(lsa_word_topic, query_vector), row_norms)
    
    return sorted(zip(cosines, vocab), reverse=True)
    
def closest_docs(query_index):
    row_norms = np.sqrt(np.sum(lsa_doc_topic**2, axis=1))
    query_vector = lsa_doc_topic[query_index]
    cosines = np.divide(np.dot(lsa_doc_topic, query_vector), row_norms)
    
    return sorted(zip(cosines, [novel["title"] for novel in novels]), reverse=True)
    
def topic_words(col):
    return sorted(zip(lsa_word_topic[:,col], vocab), reverse=True)

def categories_to_numbers(list_of_categories):
    distinct_values = list(set(list_of_categories))
    category_numbers = []
    for category in list_of_categories:
        category_numbers.append(distinct_values.index(category))
    return category_numbers
    
def novel_token_counts():
    return np.sum(novelmatrix, axis=0)

if __name__ == '__main__':
    # We'll run LSA with a reduction to 20 dimensions. What happens if you use
    # more?
    dimension = 20
    # We're also providing the option to throw away the first K words of the
    # vocabulary, but we're starting it at 0.
    word_offset = 0
    vocab, novels, novelmatrix = read_novels()
    
    # first 10 elements of the first column
    #  lsa_doc_topic[:10,0]
    
    lsa_word_topic, lsa_doc_topic = lsa(novelmatrix[:,word_offset:], dimension)
    
    print [(novel["id"], novel["Title"]) for novel in novels]
    
    #numpy arrays can do this thing with the ranges
    #colon comma zero is the first column of the lsa_doc_topic array which are novels
    #y dimension is the same thing but the second column
    #c is the code for colors
    #for each novel in teh list of novels add the value for author gender
    #from the metadata file
    #change the m and f into numbers because python knows how to change
    #the numbers into colors but not the letters
    
    #alpha is opacity (rgba)
    #first dimension is just the vocabulary list, so the second one is
    #where meaningish stuff starts
    matplotlib.pyplot.scatter(lsa_doc_topic[:,0], lsa_doc_topic[:,1], c=categories_to_numbers([novel["Number of Volumes"] for novel in novels]), alpha=0.5)
