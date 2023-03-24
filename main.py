from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords') 
app = Flask(__name__)
cors = CORS(app)
stop_words = ''
s = ""

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

def preprocess(DOCUMENT):
    DOCUMENT = re.sub(r'\n|\r', ' ', DOCUMENT)
    DOCUMENT = re.sub(r' +', ' ', DOCUMENT)
    DOCUMENT = DOCUMENT.strip()
    sentences = nltk.sent_tokenize(DOCUMENT)
    stop_words = nltk.corpus.stopwords.words('english')
    normalize_corpus = np.vectorize(normalize_document)
    norm_sentences = normalize_corpus(sentences)
    return sentences, norm_sentences

@app.route('/', methods=['GET', 'POST'])
def index():
    print("loaded...")
    print(request.method)
    if request.method == "POST":
        global s
        s = ""
        text = request.form.get("txt_input")
        if len(text) < 10:
            return "StringERR"

        result = preprocess(text)
        tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
        dt_matrix = tv.fit_transform(result[1])
        dt_matrix = dt_matrix.toarray()

        vocab = tv.get_feature_names_out()
        td_matrix = dt_matrix.T
        #print(td_matrix.shape)
        pd.DataFrame(np.round(td_matrix, 2), index=vocab).head(10)

        from scipy.sparse.linalg import svds
            
        def low_rank_svd(matrix, singular_count=2):
            u, s, vt = svds(matrix, k=singular_count)
            return u, s, vt

        num_sentences = 8
        num_topics = 3

        u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)  
        #print(u.shape, s.shape, vt.shape)
        term_topic_mat, singular_values, topic_document_mat = u, s, vt

        # remove singular values below threshold                                         
        sv_threshold = 0.5
        min_sigma_value = max(singular_values) * sv_threshold
        singular_values[singular_values < min_sigma_value] = 0

        salience_scores = np.sqrt(np.dot(np.square(singular_values), 
                                        np.square(topic_document_mat)))
        salience_scores

        top_sentence_indices = (-salience_scores).argsort()[:num_sentences]
        top_sentence_indices.sort()
        print(top_sentence_indices)
        print(result[0])
        summary = '\n'.join(np.array(result[0])[top_sentence_indices])
         
        return summary

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
