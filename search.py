#!/usr/bin/python3
import os
import nltk
import sys
import getopt
import json
import numpy as np
from nltk.stem.porter import PorterStemmer

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

STEMMER = PorterStemmer()
DICTIONARY = {}


class Query:

    def __init__(self, query, tokens , counts, is_phrase):
        self.query = query
        self.is_phrase = is_phrase
        self.tokens = tokens
        self.counts = counts
        self.query_weights = 0
        self.relevant_docs = []
        
class Posting:
    
    def __init__(self, term, tf , docs):
        self.term = term
        self.tf = tf
        # 
        # doc_id: postions => {1:[1,3,6], 2:[12], 5:[9,19]}
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
        open(postings_file, mode="rb") as postings_file:
            
        # load dictionary and length into memory
        DICTIONARY = json.load(dictionary_file)
        
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
        
        # query = refine_query(parsed_queries)
        
        result = []
        # result = evaluate_query(parsed_queries , postings_file, relevant_docs, posting_file, N)
            
        # write result into file
        result = map(str, result)
        result = ' '.join(result)
        fout.write(result)

def refine_query(query):
    # 1. spell correction
    # 2. expand query
    # 3. pseudo relevant feedback 
    
    return []

def evaluate_query(queries , postings_file, relevant_docs, posting_file, N):
    # 2.1 get all relevant documents for each subquery
    # for subquery in queries
    #     if subquery is phrase query
    #         docs = SearchPhraseQueryOnContent(query) 
    #     else subquery is free text query
    #         docs = SearchFreeTextQueryOnContent(query) HW3
        
    # 2.2 intersection

    # 2.3 caculate the score
    # for subquery in queries
    #     for each token t in subquery
    #         doc_weight = posting_list[t].tf
    #         scores[doc_id] += subquery.query_weight * doc_weight

        
    # for each doc_id in scores
    #     scores[doc_id]=scores[doc_id]/Length(doc_id) 

    # 2.4 rank  
    return []
    
    
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
        if subquery[0] == '"' and subquery[-1] == '"':
            tokens, count = tokenize_query(subquery[1:-1])
            queries.append(Query(subquery[1:-1], tokens, count, True))
        else:
            tokens, count = tokenize_query(subquery)
            queries.append(Query(subquery, tokens, count, False))

    return queries


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
    Get posting list and tfs for given token.

    Args:
        token (str): A token searching for.
        postings_file (BufferedReader): File reader for posting file.

    Returns:
        Posting list and tfs for given token.
    """
    return [], {}

    



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
