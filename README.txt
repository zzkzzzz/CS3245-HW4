This is the README file for A0220882H-A0220910X-A0267777E-??????'s submission
Emails: e0556074@u.nus.edu, e0556102@u.nus.edu, e1100297@u.nus.edu

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

8. Then we write the data into postings.txt and dictionary.txt

Form:

dictionary: term: (length of posting, offset)

postings: term: term: (doc_id: (tf, positions)), (doc_id: (tf, positions))....


----- Searching -----
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

(However, in the final version, we disable query word correction and query expansion as the performance does not incease)

2. Query evaluation

3. Write the result to result file.





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

[x] I/We, A0220882H and A0220910X, certify that I/we have followed the CS 3245 Information
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
