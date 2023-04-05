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
from os import remove
import csv
import pickle

csv.field_size_limit(500 * 1024 * 1024)


# python index.py -i dataset.csv -d dictionary.txt -p postings.txt


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
	
	for token in tokens:
		token = stemmer.stem(token).lower() #stemming and case-folding
	return tokens

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
	
	with open(in_csv, newline='') as f:
		reader = csv.reader(f, dialect='excel')
		doc_num = 1
		#print(reader.__sizeof__())
		#l =[]
		#for i in reader:
			#l.append(i)
		#print(len(l))
		#exit()
		for i in reader:
			if i[0] == 'document_id':
				continue
			
			doc_id, date, title, content, court = i[0], [i[3]], preprocess(i[1]), preprocess(i[2]), preprocess(i[4])
			
			#print(doc_num)
			#print(i)
			#print('index title...')
			# deal with title
			doc_termFreq = {}   # like: {term: freq}
			doc_termPositions = {}  # like: {term: [1, 2, 3, 4]}
			for posi, term in enumerate(title):
				if term not in doc_termFreq:
					doc_termFreq[term] = 1
					doc_termPositions[term] = [posi]
				else:
					doc_termFreq[term] = doc_termFreq[term] + 1
					doc_termPositions[term].append(posi)
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
				if term not in positional_index['title']:
					positional_index['title'][term] = {doc_id: [log_tf/sum, doc_termPositions[term]]}
				else:
					positional_index['title'][term][doc_id] = [log_tf/sum, doc_termPositions[term]]
					
			#print('index content...')
			# deal with content
			doc_termFreq = {}   # like: {term: freq}
			doc_termPositions = {}  # like: {term: [1, 2, 3, 4]}
			for posi, term in enumerate(content):
				if term not in doc_termFreq:
					doc_termFreq[term] = 1
					doc_termPositions[term] = [posi]
				else:
					doc_termFreq[term] = doc_termFreq[term] + 1
					doc_termPositions[term].append(posi)
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
				if term not in positional_index['content']:
					positional_index['content'][term] = {doc_id: [log_tf/sum, doc_termPositions[term]]}
				else:
					positional_index['content'][term][doc_id] = [log_tf/sum, doc_termPositions[term]]
			
			#print('index date...')
			# deal with date
			doc_termFreq = {}   # like: {term: freq}
			doc_termPositions = {}  # like: {term: [1, 2, 3, 4]}
			for posi, term in enumerate(date):
				if term not in doc_termFreq:
					doc_termFreq[term] = 1
					doc_termPositions[term] = [posi]
				else:
					doc_termFreq[term] = doc_termFreq[term] + 1
					doc_termPositions[term].append(posi)
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
				if term not in positional_index['date']:
					positional_index['date'][term] = {doc_id: [log_tf/sum, doc_termPositions[term]]}
				else:
					positional_index['date'][term][doc_id] = [log_tf/sum, doc_termPositions[term]]
			
			#print('index court...')
			# deal with court
			doc_termFreq = {}   # like: {term: freq}
			doc_termPositions = {}  # like: {term: [1, 2, 3, 4]}
			for posi, term in enumerate(court):
				if term not in doc_termFreq:
					doc_termFreq[term] = 1
					doc_termPositions[term] = [posi]
				else:
					doc_termFreq[term] = doc_termFreq[term] + 1
					doc_termPositions[term].append(posi)
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
				if term not in positional_index['court']:
					positional_index['court'][term] = {doc_id: [log_tf/sum, doc_termPositions[term]]}
				else:
					positional_index['court'][term][doc_id] = [log_tf/sum, doc_termPositions[term]]
			if doc_num % 100 == 1:
				print(doc_num)
			doc_num += 1
					
	print('writing to file....')
	dictionary = {}
	dictionary['title'] = {}
	dictionary['content'] = {}
	dictionary['date'] = {}
	dictionary['court'] = {}
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
	for term, posting in positional_index['content'].items():
		pik_posting = pickle.dumps(posting)
		pick_length = len(pik_posting)
		
		dictionary['content'][term] = (len(posting), offset)
		offset += pick_length
		postings += pik_posting
	
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
	
	with open(out_postings, 'wb') as pos:
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