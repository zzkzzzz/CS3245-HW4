#!/usr/bin/env python3

import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import nltk
import sys
import getopt
import json
import math
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from os import remove
import csv
import pickle
import re
import string
import time
from datetime import datetime
csv.field_size_limit(500 * 1024 * 1024)


# python index.py -i dataset.csv -d dictionary.txt -p postings.txt

def deal_zone(zone, posidex, doc_id1):
	doc_id1 = int(doc_id1)
	# deal with title
	doc_termFreq = {}   # like: {term: freq}
	doc_termPositions = {}  # like: {term: [1, 2, 3, 4]}
	posi_index = posidex.copy()
	# count = 0
	for posi, term in enumerate(zone):
		if term not in doc_termFreq:
			doc_termFreq[term] = 1
			doc_termPositions[term] = [posi]
		else:
			doc_termFreq[term] = doc_termFreq[term] + 1
			doc_termPositions[term].append(posi)
		# count += 1
	# log tf
	for term, tf in doc_termFreq.items():
		doc_termFreq[term] = logg_tf(tf)
	# doc_length
	sum = 0
	for term, tf in doc_termFreq.items():
		sum += math.pow(tf, 2)
	sum = math.sqrt(sum)
	# positional_index: term, doc_id: [tf, positions]
	for term, log_tf in doc_termFreq.items():
		if term not in posi_index:
			posi_index[term] = {doc_id1: [log_tf/sum, doc_termPositions[term]]}
		else:
			posi_index[term][doc_id1] = [log_tf/sum, doc_termPositions[term]]
	return posi_index		
	# return posi_index, count

			
def usage():
	print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")
	
def logg_tf(tf):
	if tf > 0:
		return 1 + math.log(tf, 10)
	else:
		return 0
	
def preprocess(text):
	stemmer = PorterStemmer() # initialize stemmer
	sentences = nltk.sent_tokenize(text)
	tokenized_sent = [nltk.word_tokenize(s) for s in sentences] # still array of tokenized sentences
	tokens = [word for sent in tokenized_sent for word in sent] # flatten array
	tok = []
	for token in tokens:
		if len(token) != 0:
			if token[0] in string.punctuation:
				token = token[1:]
		if len(token) != 0:
			if token[-1] in string.punctuation:
				token = token[:-1]
		token = stemmer.stem(token).lower() #stemming and case-folding
		if token not in string.punctuation:
				tok.append(token)
	return tok

def build_index(in_csv, out_dict, out_postings):
	"""
	build index from documents stored in the input directory,
	then output the dictionary file and postings file
	"""
	print('indexing...')
	positional_index = {}
	positional_index['title'] = {}
	positional_index['content'] = {}
	positional_index['date'] = {}
	positional_index['court'] = {}
	start = time.process_time()
	with open(in_csv, newline='') as f:
		reader = csv.reader(f, dialect='excel')
		doc_num = 0
		lengths = 0
		for i in reader:
			doc_num += 1
			if i[0] == 'document_id':
				continue
			# if doc_num > 200:
			# 	break
			doc_id, date, title, content, court = i[0], [i[3]], preprocess(i[1]), preprocess(i[2]), preprocess(i[4])
			
			positional_index['title'] = deal_zone(title, positional_index['title'], doc_id)
			positional_index['content'] = deal_zone(content, positional_index['content'], doc_id)
			# positional_index['content'], length = deal_zone(content, positional_index['content'], doc_id)
			positional_index['date'] = deal_zone(date, positional_index['date'], doc_id)
			positional_index['court'] = deal_zone(court, positional_index['court'], doc_id)
			
			if doc_num % 100 == 1:
				print('Time taken till now: ' + str(time.process_time() - start) + 's')
				print(doc_num)
			lengths += length
		# print("average_doc_length is " + str(lengths/17154))
				
	print('writing to file....')
	dictionary = {}
	dictionary['title'] = {}
	dictionary['content'] = {}
	dictionary['date'] = {}
	dictionary['court'] = {}
	dictionary[''] = doc_num
	postings = b''
	offset = 0
	
	print('post title...')
	for term, posting in positional_index['title'].items():
		pik_posting = pickle.dumps(posting)
		pick_length = len(pik_posting)
		
		dictionary['title'][term] = (len(posting), offset)
		offset += pick_length
		postings += pik_posting
	
	print('post content...')
	print(str(len(positional_index['content'].items())) + ' unique terms')
	count = 0
	start = time.process_time()
	for term, posting in positional_index['content'].items():
		pik_posting = pickle.dumps(posting)
		pick_length = len(pik_posting)
		
		dictionary['content'][term] = (len(posting), offset)
		offset += pick_length
		postings += pik_posting
		#print(len(postings))
		if (len(postings) > 10000000):
			pos = open(out_postings, 'ab')
			pos.write(postings)
			postings = b''
		if count %1000 == 0:
			print(str(count) + " done in " + str(time.process_time()-start) + 's')
		count += 1
		
	print('post date...')
	for term, posting in positional_index['date'].items():
		pik_posting = pickle.dumps(posting)
		pick_length = len(pik_posting)
		
		dictionary['date'][term] = (len(posting), offset)
		offset += pick_length
		postings += pik_posting
		
	print('post court...')
	for term, posting in positional_index['court'].items():
		pik_posting = pickle.dumps(posting)
		pick_length = len(pik_posting)
		
		dictionary['court'][term] = (len(posting), offset)
		offset += pick_length
		postings += pik_posting
		
		
	print('writing...')
	
	with open(out_postings, 'ab') as pos:
		pos.write(postings)
		
	with open(out_dict, 'wb') as dic:
		pickle.dump(dictionary, dic)
		
	print('done')
	
	
input_directory = output_file_dictionary = output_file_postings = None

try:
	opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
	usage()
	sys.exit(2)
	
for o, a in opts:
	if o == '-i': # input directory
		input_directory = a
	elif o == '-d': # dictionary file
		output_file_dictionary = a
	elif o == '-p': # postings file
		output_file_postings = a
	else:
		assert False, "unhandled option"
		
if input_directory == None or output_file_postings == None or output_file_dictionary == None:
	usage()
	sys.exit(2)
	
build_index(input_directory, output_file_dictionary, output_file_postings)