# Requirements
- Vertical search engine like google scholar
- Limited to Papers/books published by member of Research Centre for Intelligent Healthcare (RCIH)
- Extract authors, publication year and title
- Once a week, crawler schedule
- Interface like google scholar
- query/keywords based; arranged by relevancy
- Web based
- Subject classification


More:
Fully working crawler component
Construction of inverted index
Fully working query processor component
fully working subject classification component
Overall usability


the system is expected to be working search engine with reasonable accuracy and speed. This ensures that the system contains fully working crawler and query processor components. In addition, it must have at least one, and preferably both, of the other two components, i.e. the inverted index and the text classification components, in fully working status.


# Process:

- Document Collection: Start with a collection of documents that you want to index and make searchable. These documents can be in various formats such as text files, web pages, or any other structured/unstructured data.

- Preprocessing: Clean and preprocess the documents to remove any irrelevant information, such as HTML tags, punctuation, and stop words (common words like "the," "is," etc.). Additionally, perform other text normalization techniques like stemming or lemmatization to reduce words to their base form.

- Inverted Index Construction: Build an inverted index, which is a data structure that maps each unique term in the document collection to a list of document identifiers (IDs) where the term occurs. The inverted index speeds up the process of finding relevant documents for a given query. For each term, store the list of documents containing that term along with additional information like term frequency (TF) or inverse document frequency (IDF).

- Text Classification: Implement a text classification component to categorize or assign labels to the documents based on their content. This step is optional but can enhance the retrieval system's functionality. You can use various techniques for text classification, such as supervised machine learning algorithms like Naive Bayes, Support Vector Machines (SVM), or deep learning models like Convolutional Neural Networks (CNN) or Recurrent Neural Networks (RNN).

- User Query Processing: When a user submits a query, preprocess the query in a similar way to the document preprocessing step. Remove stop words, punctuation, and perform any necessary normalization. Then, use the inverted index to quickly identify relevant documents. Retrieve the list of document IDs for each query term and calculate a relevance score for each document based on different ranking algorithms like TF-IDF, BM25, or PageRank.

- Document Retrieval and Ranking: Rank the retrieved documents based on their relevance scores and present the most relevant documents to the user. You can use ranking algorithms like the Vector Space Model or Okapi BM25 to determine the document ranking.

- Presentation and User Interface: Finally, present the search results to the user through an intuitive user interface that allows them to explore the retrieved documents. This can include displaying snippets or summaries of the documents, highlighting the query terms in the document text, or providing additional filtering and sorting options.


## Process
- Document Collection: 
Start with a collection of documents that you want to index and make searchable. These documents can be in various formats such as text files, web pages, or any other structured/unstructured data.

- Preprocessing: 
Apply text preprocessing techniques such as tokenization, lowercase conversion, stop word removal, and lemmatization to the documents and the search query. This ensures that they are represented consistently and without noise.

- Document vector representations:
Use a suitable vectorization technique, such as TF-IDF, to convert the preprocessed documents and the search query into numerical vector representations. This step transforms the text into a format that can be used for cosine similarity calculations.

- Calculate the cosine similarity between the search query and each document:
Compute the cosine similarity between the vector representation of the search query and each document using the cosine similarity formula. The higher the cosine similarity score, the more similar the document is to the search query.

- Rank the documents based on cosine similarity scores:
Sort the documents in descending order of their cosine similarity scores. The document with the highest cosine similarity score is considered the most relevant to the search query.
