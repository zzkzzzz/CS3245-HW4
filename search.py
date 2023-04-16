#!/usr/bin/python3
import os
import nltk
import sys
import getopt
import numpy as np
from nltk.stem.porter import PorterStemmer
from refine import correct_query, expand_query
from utils import idf, tf


try:
    import cPickle as pickle
except ImportError:
    import pickle
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# python3.8 search.py -d dictionary.txt -p postings.txt -q queries/q1.txt  -o result.txt

STEMMER = PorterStemmer()
DICTIONARY = {}
POSTINGS = None

N = 17000 # replace with number of documents in collection

class Query:

    def __init__(self, query, tokens , counts, is_phrase):
        self.query = query
        self.is_phrase = is_phrase
        self.tokens = tokens
        self.counts = counts
        self.query_weights = {}
        self.relevant_docs = []
        
class Posting:
    
    def __init__(self, term, docs):
        self.term = term
        # doc_id: postions => {1:[tf,[1,3,6]], 5:[tf,[9,19]]}
        self.docs = docs
    


def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")


def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')
    
    with open(queries_file, 'r') as fin, \
        open(results_file, 'w') as fout, \
        open(dict_file, mode="rb") as dictionary_file, \
        open(postings_file, mode="rb") as postings_files:
            
        global DICTIONARY
        DICTIONARY = pickle.load(dictionary_file)

        global POSTINGS
        POSTINGS = postings_files
       
        query_str = ""
        relevant_docs = []
        
        first_line = True
        for line in fin:
            if first_line:
                query_str = line
                first_line = False
            else:
                relevant_docs.append(int(line))

        parsed_queries = parse_query(query_str)  
        
        result = []
        # result = evaluate_query(parsed_queries , postings_file, relevant_docs, posting_file, N)
        result = evaluate_query(parsed_queries, relevant_docs, N)
        # write result into file
        result = map(str, result)
        result = ' '.join(result)
        fout.write(result)

def refine_query(query):
    query = correct_query(query)
    query = expand_query(query)
    
    return query

def search_two_word_phrase(words):
    result = []
    idx1, idx2 = 0, 0
    # get posting object
    postings_lst1 = get_postings(words[0], POSTINGS)
    postings_lst2 = get_postings(words[1], POSTINGS)
    # get doc IDs in posting list
    docs1 = sorted(postings_lst1.docs.keys())
    docs2 = sorted(postings_lst2.docs.keys())

    while idx1 < len(docs1) and idx2 < len(docs2):
        if docs1[idx1] == docs2[idx2]:
            # words appearing in the same document, check position
            pos1 = postings_lst1.docs[docs1[idx1]][1]
            pos2 = postings_lst2.docs[docs2[idx2]][1]
            # check if two word1 immediately comes before word2
            i, j = 0, 0
            while i < len(pos1) and j < len(pos2):
                if pos1[i] == pos2[j] - 1:
                    # phrase found
                    result.append(docs1[idx1])
                    break
                elif pos1[i] >= pos2[j]:
                    j += 1
                else:
                    i += 1
            idx1 += 1
            idx2 += 1
        elif docs1[idx1] < docs2[idx2]:
            idx1 += 1
        else:
            idx2 += 1
    return result

def search_three_word_phrase(words):
    result = []
    idx1, idx2, idx3 = 0, 0, 0
    # get posting object
    postings_lst1 = get_postings(words[0], POSTINGS)
    postings_lst2 = get_postings(words[1], POSTINGS)
    postings_lst3 = get_postings(words[2], POSTINGS)
    # get doc IDs in posting list
    docs1 = sorted(postings_lst1.docs.keys())
    docs2 = sorted(postings_lst2.docs.keys())
    docs3 = sorted(postings_lst3.docs.keys())

    while idx1 < len(docs1) and idx2 < len(docs2) and idx3 < len(docs3):
        # all 3 words appearing in the same doc
        if docs1[idx1] == docs2[idx2] and docs2[idx2] == docs3[idx3]:
            # check if the 3 words appears consecutively
            pos1 = postings_lst1.docs[docs1[idx1]][1]
            pos2 = postings_lst2.docs[docs2[idx2]][1]
            pos3 = postings_lst3.docs[docs3[idx3]][1]

            i, j, k = 0, 0, 0
            while i < len(pos1) and j < len(pos2) and k < len(pos3):
                if pos1[i] == pos2[j] - 1 and pos1[i] == pos3[k] - 2:
                    result.append(docs1[idx1])
                    break
                elif pos1[i] >= pos2[j]:
                    j += 1
                elif pos1[i] >= pos3[k] or pos2[j] >= pos3[k]:
                    # after incrementing 2nd position list have to check 
                    # if both 1st and 2nd position is before 3rd position
                    k += 1
                else:
                    i += 1 
            pass
        elif docs1[idx1] <= docs2[idx2] and docs1[idx1] <= docs3[idx3]:
            # docID1 <= docID 2 and docID1 <= docID 3
            idx1 += 1
        elif docs1[idx1] >= docs2[idx2] and docs2[idx2] <= docs3[idx3]:
            # docID1 >= docID 2 and docID2 <= docID 3
            idx2 += 1
        else:
            idx3 += 1
    return result

def search_phrase_on_content(query):
    """
    TODO positional search
    self.query = query
    self.is_phrase = is_phrase
    self.tokens = tokens
    self.counts = counts
    self.query_weights = 0
    self.relevant_docs = []
    """
    words = query.tokens
    num_words = len(words)
    if num_words == 1:
        postings_lst = get_postings(words[0], POSTINGS)
        return postings_lst.docs.keys()
    elif num_words == 2:
        return search_two_word_phrase(words)
    elif num_words == 3:
        return search_three_word_phrase(words)
    return []

def score_documents(query):
    scores = {}
    terms = query.counts.keys()
    weights = query.query_weights
    for term in terms:
        if term not in DICTIONARY['content']:
            continue
        # print(token)
        postings_lst = get_postings(term, POSTINGS).docs
        docs = postings_lst.keys()
        query_weight = weights[term]
        for docID in docs:
            doc_weight = postings_lst[docID][0]
            if docID in scores:
                scores[docID] += query_weight * doc_weight
            else:
                scores[docID] = query_weight * doc_weight
    return sorted(scores.keys(), key=lambda x:x[1])

def evaluate_query(queries, relevant_docs, N):
    """
    No need relevant_docs here?? since it is given along with query as a kind of feedback, would be used for query refinement
    """
    # 2.1 get all relevant documents for each subquery
    # for subquery in queries
    #     if subquery is phrase query
    #         docs = SearchPhraseQueryOnContent(query) 
    #     else subquery is free text query
    #         docs = SearchFreeTextQueryOnContent(query) HW3
    candidate_docs = [] # stores relevant document IDs for free text query and related documents IDs for phrase query
    for subquery in queries:
        if subquery.is_phrase:
            docs = search_phrase_on_content(subquery)
        else:
            docs = score_documents(subquery) # homework 3
        candidate_docs.append(docs)
    # 2.2 intersection
    candidate_docs = [set(x) for x in candidate_docs]
    final_docs = set.intersection(*candidate_docs)

    # 2.3 caculate the score
    # for subquery in queries
    #     for each token t in subquery
    #         doc_weight = posting_list[t].tf
    #         scores[doc_id] += subquery.query_weight * doc_weight
    scores = {}
    for subquery in queries:
        terms = subquery.counts.keys()
        weights = subquery.query_weights
        for term in terms:
            if term not in DICTIONARY['content']:
                continue
            postings_lst = get_postings(term, POSTINGS).docs
            docs = set(postings_lst.keys())
            docs = set.intersection(final_docs, docs)
            query_weight = weights[term]
            for docID in docs:
                doc_weight = postings_lst[docID][0]
                if docID in scores:
                    scores[docID] += query_weight * doc_weight
                else:
                    scores[docID] = query_weight * doc_weight
    # normalize
    """
    Don't need the normalize step? already calcuated when constructing posting list?
    """
    # for doc_id in scores:
    #     scores[doc_id] = scores[doc_id]/LENGTH[str(doc_id)]

    # 2.4 rank
    # sort doc id by scores
    # print(sorted(scores.items(), key=lambda x:x[1], reverse=True))
    return sorted(scores, key=scores.get, reverse=True)
    
    
def parse_query(query):
    """
    Parse a query string into a list of Query object.

    Args:
        query (str): A string query

    Returns:
        A list of Query object.
    """
    queries = []

    subqueries = query.split('AND')
    for subquery in subqueries:
        subquery = subquery.strip()
        # phrase query is surrounded by double quotes
        # no query expansion for phrase query
        if subquery[0] == '"' and subquery[-1] == '"':
            tokens, count = tokenize_query(subquery[1:-1])
            queries.append(Query(subquery[1:-1], tokens, count, True))
        # free text query
        else:
            subquery = refine_query(subquery)
            tokens, count = tokenize_query(subquery)
            queries.append(Query(subquery, tokens, count, False))
            
    caculate_query_weight(queries)
    
    return queries

def caculate_query_weight(queries):
    for query in queries:
        for token in query.counts.keys():
            if token not in DICTIONARY["content"]:
                continue
            query.query_weights[token] = tf(query.counts[token]) * idf(N, DICTIONARY["content"][token][0])
    

def tokenize_query(query):
    """
    Tokenize and stem a string into token list.

    Args:
        query (str): A query given by user.

    Returns:
        A list of token.
    """
    # tokenize the query string
    tokens = [word for sent in nltk.sent_tokenize(query)
                       for word in nltk.word_tokenize(sent)]
    
    # TODO: remove stop words?

    # case folding and stem the tokens
    result = []
    count = {}
    for token in tokens:
        new_token = STEMMER.stem(token.lower())
        result.append(new_token)
        if new_token in count:
            count[new_token] = count[new_token] + 1
        else:
            count[new_token] = 1

    return result, count


def get_postings(token, postings_file):
    """
    Get posting list for given token.

    Args:
        token (str): A token searching for.
        postings_file (BufferedReader): File reader for posting file.

    Returns:
        Posting list for given token.
    """

    postings_file.seek(DICTIONARY["content"][token][1])

    return Posting(token,pickle.load(postings_file))

    



dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None:
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
