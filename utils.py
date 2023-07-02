import pandas as pd
import json

import requests
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def tokenizer(statement):
    # Filter out stop words & special characters
    stop_words = set(stopwords.words('english'))
    special_characters = '''!()-—[]{};:'"\, <>./?@#$%^&*_~+='''
    tokens = word_tokenize(statement)
    return [token.lower() for token in tokens if token.lower() not in stop_words and token not in special_characters]

def lemmatize_word(word):
    wordnet_tags = {"V": wordnet.VERB, "R": wordnet.ADV,"N": wordnet.NOUN,"J": wordnet.ADJ} 
    # Get parts of speech tag & determine the class in wordnet
    pos_tag = nltk.pos_tag([word])[0][1][0].upper()
    pos_tag_class = wordnet_tags.get(pos_tag, wordnet.NOUN)
    lemmatizer = WordNetLemmatizer()
    # Lemmaitze with Part of Speech Tag to get the pure word
    lemma = lemmatizer.lemmatize(word, pos=pos_tag_class)
    return lemma

def lemmatize_stmt(statement):    
    filtered_tokens = tokenizer(statement)
    lemmatize_tokens = []
    for word in filtered_tokens:
        lemma = lemmatize_word(word)
        lemmatize_tokens.append(lemma)
    return ' '.join(lemmatize_tokens)

def scrape_pureportal():
    print("Scraping")
    BASE_URL = 'https://pureportal.coventry.ac.uk/'
    RCIH_publications = BASE_URL + 'en/organisations/centre-for-intelligent-healthcare/publications/'
    research_output = []
    url_path = RCIH_publications
    while url_path is not None:
        print(".", end="")
        response = requests.get(url_path)
        soup = BeautifulSoup(response.content, "html.parser")
        li_item_tags = soup.find_all('li', class_= 'list-result-item')
        for li_item in li_item_tags:
            research_link = li_item.find('a')['href']
            authors = [{'author':author.text, 'url': author['href']} for author in li_item.findAll('a', 'link person')]
            published_date = li_item.find('span', class_='date').text
            title = li_item.find('h3', class_='title').text
            categories = [concept.text for concept in li_item.findAll('span', class_ = 'concept')]
            imp = lemmatize_stmt(title)  + ' ' + \
                " ".join([author['url'].split('/')[-1].replace('-', ' ') for author in authors]) + ' ' + \
                    lemmatize_stmt(" ".join(categories)) + ' ' + \
                lemmatize_stmt(published_date)
            research_output.append({"authors":authors, "published_date":published_date, "title":title, "research_link": research_link, "categories": categories, "imp": imp})
        nextLinkTag = soup.find('a', class_ = 'nextLink')
        if nextLinkTag is not None:
            url_path = BASE_URL+nextLinkTag['href']
        else:
            url_path = None
    print('\nTotal Document Scrapped:', len(research_output))
    
    # Dump scraped data to json file
    with open('./scraped_data/rcih_research_output.json', 'w') as f:
        json.dump(research_output, f, indent=4)
    
    # Inverted Index Construction
    inverted_index = {}
    for doc_id, doc in enumerate(research_output):
        imp = lemmatize_stmt(doc['title']  + ' ' + " ".join(doc['categories']) + ' ' + doc['published_date']) + ' ' + \
                " ".join([author['url'].split('/')[-1].replace('-', ' ') for author in doc['authors']])
        for term in imp.split():
            if term not in inverted_index:
                inverted_index[term] = set()
            inverted_index[term].add(doc_id)

    # Dump Inverted Index data to json file
    with open('./scraped_data/inverted_index.json', 'w') as f:
        for key in inverted_index:
            inverted_index[key] = list(inverted_index[key])
        json.dump(inverted_index, f, indent=4)


def search_engine(query,skip_result=0, limit_result=10):
    lemmatized_query = [lemmatize_stmt(query)]

    with open('./scraped_data/inverted_index.json') as f:
        inverted_index = json.loads(f.read())
    with open('./scraped_data/rcih_research_output.json') as f:
        rcih_research = json.loads(f.read())
    
    token_docs_set = set()
    for query_token in lemmatized_query[0].split():
        try:
            token_docs = inverted_index[query_token]
            token_docs_set.update(token_docs)
        except:
            pass
    sorted_token_docs_set = sorted(token_docs_set)

    processed_research = []
    for doc_id in sorted_token_docs_set:
        processed_research.append(rcih_research[doc_id])

    df = pd.DataFrame.from_dict(processed_research)

    documents = []
    try:
        vectorizer = TfidfVectorizer()
        document_vectors = vectorizer.fit_transform(df['imp'])
        query_vector = vectorizer.transform(lemmatized_query)
        cosine_similarities = cosine_similarity(query_vector, document_vectors).flatten()
        sorted_indices = cosine_similarities.argsort()[::-1]
        for i in range(limit_result):
            index_with_skip = i+skip_result
            if(index_with_skip < len(sorted_token_docs_set)):
                index = sorted_indices[index_with_skip]
                relevant_document = df.iloc[index].to_dict()
                relevant_document["relevance_score"] = round(cosine_similarities[index] * 100,2)
                documents.append(relevant_document)
    except:
        pass
    return documents

def classifier(test_texts):
    training_news_df = pd.read_pickle('training_news_df.pkl')
    texts = training_news_df['short_description'].tolist()
    labels = training_news_df['category'].tolist()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    classifier = MultinomialNB()
    X_train, _, y_train, _ = train_test_split(X, labels, test_size=0.2, random_state=42)
    classifier.fit(X_train, y_train)
    test_X = vectorizer.transform([test_texts])
    test_predictions = classifier.predict(test_X)
    return {"result": test_predictions[0]}