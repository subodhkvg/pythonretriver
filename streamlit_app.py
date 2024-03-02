from flask import Flask,render_template,request
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Load them back
from joblib import dump, load
import json

app = Flask(__name__)
@app.route('/')

def index():
    return "Hello World"

@app.route('/predict',methods=['GET'])

def predict_placement():
    query = [request.args.get('query')]
    #cgpa = "2"
    #print(cgpa)
    authors_vectors = load('authors_vectors.joblib')
    pmcid_vectors = load('pmcid_vectors.joblib')
    doc_vectors = load('doc_vectors.joblib')
    pmid_vectors = load('pmid_vectors.joblib')
    
    vectorizer = load('vectorizer.joblib')
    pmid_vectoizer = load('pmid_vectoizer.joblib')
    doc_vectoizer = load('doc_vectoizer.joblib')
    pmcid_vectoizer = load('pmcid_vectoizer.joblib')

    author_documents = load('author_documents.joblib')
    doc_context = load('doc_context.joblib')  
    links = load('links.joblib') 

    print(links[19])
    print(links[20])
    print(links[21])

    top_n = 3
    counter = 0
    matched_documents = {}
    all_matched_documents = {} 

    #query = ["articles related to 35024486"]
    
    #query = ["titles of author yang"]
    
    #.................................................
    #Case 1: Check if Question is related to author
    #.................................................
    print(f"Author\n")  
    query_vector = vectorizer.transform(query)  
    similarity_scores = cosine_similarity(query_vector, authors_vectors)
    sorted_doc_indices = similarity_scores.argsort().flatten()[::-1]
    for index in sorted_doc_indices[:top_n]:
        #print(f"Doc {index}: {author_documents[index]} with score {similarity_scores[0][index]}")
        if float(similarity_scores[0][index]) > 0.00:
            print(f"Doc {index}: {author_documents[index]} with score {similarity_scores[0][index]}")
            matched_documents = {}
            matched_documents['index'] = str(index)
            matched_documents['match'] = author_documents[index]
            matched_documents['score'] = similarity_scores[0][index]
            matched_documents['docs'] = doc_context[index]
            matched_documents['links'] = links[index]
            all_matched_documents[""+str(counter)+""] = matched_documents
            #print(all_matched_documents)
            counter = counter + 1
    #.................................................
    #Case 2: Check if Question is related to PMCID     
    #.................................................
    print(f"PMCID\n")
    query_vector = pmcid_vectoizer.transform(query)     
    similarity_scores = cosine_similarity(query_vector, pmcid_vectors)
    sorted_doc_indices = similarity_scores.argsort().flatten()[::-1]
    for index in sorted_doc_indices[:top_n]:
        #print(f"Doc {index}: {author_documents[index]} with score {similarity_scores[0][index]}")
        if float(similarity_scores[0][index]) > 0.00:
            print(f"Doc {index}: {author_documents[index]} with score {similarity_scores[0][index]}")
            matched_documents = {}
            matched_documents['index'] = str(index)
            matched_documents['match'] = author_documents[index]
            matched_documents['score'] = similarity_scores[0][index]
            matched_documents['docs'] = doc_context[index]
            matched_documents['links'] = links[index]
            all_matched_documents[""+str(counter)+""] = matched_documents
            counter = counter + 1

    #.................................................
    #Case 3: Check if Question is related to PMID     
    #.................................................
    print(f"PMID\n")
    query_vector = pmid_vectoizer.transform(query)
    similarity_scores = cosine_similarity(query_vector, pmid_vectors)
    sorted_doc_indices = similarity_scores.argsort().flatten()[::-1]
    for index in sorted_doc_indices[:top_n]:
        #print(f"Doc {index}: {author_documents[index]} with score {similarity_scores[0][index]}")
        if float(similarity_scores[0][index]) > 0.00:
            print(f"Doc {index}: {author_documents[index]} with score {similarity_scores[0][index]} AND INDEX {index}")
            matched_documents = {}
            matched_documents['index'] = str(index)
            matched_documents['match'] = author_documents[index]
            matched_documents['score'] = similarity_scores[0][index]
            matched_documents['docs'] = doc_context[index]
            matched_documents['links'] = links[index]
            #print(links[index])
            all_matched_documents[""+str(counter)+""] = matched_documents
            #print(all_matched_documents)
            counter = counter + 1        

    #.................................................
    #Case 4: Finally check in Abstract    
    #.................................................
    print(f"TAGS\n")
    query_vector = doc_vectoizer.transform(query)
    similarity_scores = cosine_similarity(query_vector, doc_vectors)
    sorted_doc_indices = similarity_scores.argsort().flatten()[::-1]
    for index in sorted_doc_indices[:top_n]:
        #print(f"Doc {index}: {author_documents[index]} with score {similarity_scores[0][index]}")
        if float(similarity_scores[0][index]) > 0.00:
            print(f"Doc {index}: {author_documents[index]} with score {similarity_scores[0][index]}")
            matched_documents = {}
            matched_documents['index'] = str(index)
            matched_documents['match'] = author_documents[index]
            matched_documents['score'] = similarity_scores[0][index]
            matched_documents['docs'] = doc_context[index]
            matched_documents['links'] = links[index]
            all_matched_documents[""+str(counter)+""] = matched_documents
            #print(counter)
            counter = counter + 1 

    #json_string = json.dumps(all_matched_documents)    

    sorted_data = dict(sorted(all_matched_documents.items(), key=lambda item: item[1]['score'], reverse=True))
    #print(sorted_data)

    json_string = json.dumps(sorted_data)

    #return 'subodh'
    return json_string

if __name__ == '__main__':
    app.run(debug=True)



