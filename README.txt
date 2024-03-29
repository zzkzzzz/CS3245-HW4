This is the README file for A0220882H-A0220910X-A0267777E-A0194554W's submission
Emails: e0556074@u.nus.edu, e0556102@u.nus.edu, e1100297@u.nus.edu, e0376984@u.nus.edu

== Python Version ==

We're using Python Version 3.8.10 for this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

The program consists of two major components, namely the index construction and searching.
----- Index Construction -----

1. Open csv files, store it in a reader.
2. For each file, we divided it into 4 parts (title, content, date, court).
3. We deal with these 4 parts one by one. First we do tokenize, then we remove punctuations and do stemming and case-folding. For each term, we process it and keep updating the positional_index. The structure is : positional_index: term, doc_id: [tf, positions].
4. We have 4 positional index dictionaries (for title, content, date and court).
5. Then we write them into file. Our dictionary has 5 parts: title, content, date, court and document_number.
6. For each part(title, content, date and court), we turn posting list into byte form and store it into a variable. At the same time, we also calculate the offset and length and store it in dictionary.
7. In Content part, because the posting list is too large, we store it every 10000000 length.
8. Then we write the data into postings.txt and dictionary.txt.

However, in the end, we only use index for content as incorporating information from other zones does not help a lot in our case.

Form:

dictionary: term: (length of posting, offset)

postings: term: term: (doc_id: (tf, positions)), (doc_id: (tf, positions))....


----- Searching -----
Our approach for searching is described below, which is our intended approach initially, however, we found
some of the techniques are not effective and therefore they are commented out in code. But we still describe
them here for completeness.
1. Query parsing and refinement
1.1 We will parse the query into Query objects with information of the query.

- For a boolean query, it will be split by the AND keyword and be parsed into several Query objects
- The parse_query function takes in a string query as input and returns a list of Query objects. 
It first splits the string into subqueries using the AND operator. For each subquery, it checks if 
it is a phrase query (enclosed in double quotes), and if so, tokenizes the query without any query expansion. 
If the subquery is not a phrase query, it refines the query by removing any unnecessary characters and 
then tokenizes the query using the tokenize_query function. Finally, it creates a Query object for 
each subquery and appends it to the list of queries.
- The caculate_query_weight function is then called to calculate the weight of each query, which 
is used later in the search process.

1.2 For each query object, we will do query correction(textblob) and query expansion(nltk wordnet)
- For query correction, it uses TextBlob to correct any spelling or grammar mistakes in the query.
- For query expansion, it leverages the WordNet module from NLTK to add the top 2 synonyms of each word 
in the original query to the query. This helps to broaden the search scope and potentially improve 
the relevance of the search results.

1.3 Apply relevance feedback by apply Rocchio to improve the performance of a search engine 
- The function pseudo_feedback applies the Rocchio algorithm to the query to improve its accuracy 
by using the relevant documents. The Rocchio algorithm involves multiplying the original query vector 
by a constant alpha, adding a vector composed of the average of the relevant documents' vectors weighted 
by a constant beta.

- In our implementation, the algorithm is applied to each subquery in the given query. For each 
subquery, the relevant document vectors are obtained from the postings file and used to calculate 
the new query vector. The original query vector is  multiplied by alpha, and the relevant document 
vector is normalized by the number of relevant documents and weighted by beta. The resulting vector 
is then added to the original query vector to produce the new query vector. Finally, the resulting 
query vector is returned.

- One optimization we have done is avoid getting the full relevant document vectors, instead, as long as we
accumulate more than 7000 terms from these relevant documents, we stop the process and use these terms
to modify our query. This reduces the query time while gives reasonably good performance, one can refer to
our results for sample queries in result_q1.txt, result_q2.txt, result_q3.txt, it can be seen that the given
relevant documents are given reasonably high ranks and sometimes ranked before all other documents.

In the submitted version, we disable query word correction, query expansion, and pseudo_feedback as 
the performance does not incease signficantly.

2. Query evaluation
To evaluate the query, we have two major steps. First we retrieve candidate documents
and then rank the documents by treating the whole query as a free text query.
- Iterate through all query objects. If the query object represents a phrasal query, use the
positional index to retrieve IDs of document that contains the phrase. Otherwise, process
the query as a free text query and retreive all documents with non-zero scores, and we
do not sort them yet at this step. For free text query object, we union the retrieved docIDs,
and for phrasal query object, we intersect the retrieved docID to preserve the relevance.
In the end, we union the two set of docIDs mentioned above.
- Treat the whole query as a free text query and score the set of documents obtained from previous step
using unique terms in the query. For example, if the original query is "a b" AND c d, then at this step we 
treat it as a b c d.

3. Write the result to result file.
Finally, the docIDs sorted by scores in descending order are written to file.


== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

- README.txt: Contains overview and description of the program.
- index.py: Contains main logics for building the index using SPIMI
- search.py: Contains main logics for parsing and executing queries
- refine.py: Contains main logics for refinement
- utils.py: Contains main logics for some helper functions

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[x] I/We, A0220882H, A0220910X, A0267777E and A0194554W, certify that I/we have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I/we
expressly vow that I/we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions. 

[ ] I/We, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

We suggest that we should be graded as follows:

== References ==

<Please list any websites and/or people you consulted with for this
assignment and state their role>
https://en.wikipedia.org/wiki/Shunting_yard_algorithm - for implementing the parsing algorithm
https://nlp.stanford.edu/IR-book/html/htmledition/single-pass-in-memory-indexing-1.html - for implementing indexing using SPIMI
https://www.geeksforgeeks.org/ - for usage of python functions
https://www.nltk.org/ - for the APIs used for process the raw text
