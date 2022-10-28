from rank_bm25 import BM25Okapi
from tqdm import tqdm
import string
from sklearn.feature_extraction import _stop_words
import os
import pickle
from torch import tensor


def bm25_tokenizer(text):
    tokenized_doc = []

    for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)

    return tokenized_doc


def make_bm25_docs(corpus, queries, qrels, top_k):
    corpus_ids = list(corpus.keys())
    query_texts = [ queries[qid] for qid in qrels ]
    num_queries = len(query_texts)
    passages = list(corpus.values())
    tokenized_corpus = []

    for p in passages:
        tokenized_corpus.append(bm25_tokenizer( p ))

    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = []

    print('Making bm25 scores..')
    for query in tqdm(query_texts):
        bm25_scores.append(bm25.get_scores(bm25_tokenizer(query)))

    print('Sorting scores and making the document set..')
    docset = []
    hits = 0
    doc_indeces = [False] * num_queries # Save for every query if the bm25 returns contains its relevant document.
    for i, qid in tqdm(enumerate(qrels), total=num_queries):
        bm25_score = [(score, did) for score, did in zip(bm25_scores[i], corpus_ids)]
        bm25_score.sort(reverse=True, key=lambda _:_[0])
        bm25_score = bm25_score[:top_k]
        bm25_topn_docs = [did for _, did in bm25_score]
        if qrels[qid] in bm25_topn_docs:
            hits += 1
            doc_indeces[i] = True # Check if the document id is present in the bm25 return of this query.
        docset += bm25_topn_docs

    # Construct a unique set of all documents that were returned by the top N bm25 results
    # over all query returns.
    docset = list(set(docset))
    print('bm25 docset size: ', len(docset))
    doc_texts = [corpus[did] for did in docset] # The corresponding document texts for this set.

    # Using the qrels, find for every query the matching document index
    # in the document set. If the document has not been returned by the bm25
    # for that query, the index is -1.
    print('Making the indeces..')
    for i, qid in tqdm(enumerate(qrels), total=num_queries):

        if doc_indeces[i]: # Check if for this query the document was in it's bm25 returns.
            doc_indeces[i] = docset.index( qrels[qid] ) # If so: store the index of the matching doc in the docset for this query.
        else:
            doc_indeces[i] = -1

    doc_indeces = tensor(doc_indeces, dtype=int).unsqueeze(1)

    recall = hits / num_queries

    return doc_indeces, doc_texts, recall


def disk_load_bm25_docs(corpus, queries, qrels, mode, top_k):
    # Load bm25 docs from pickle files if they exist.
    if not os.path.exists('bm25_saves/'):
        os.makedirs('bm25_saves/')

    idx_path = 'bm25_saves/' + mode + "_bm25indeces" + "_topk=" + str(top_k) + "_num-qrels=" + str(len(qrels))
    doc_path = 'bm25_saves/' + mode + "_bm25doctexts" + '_topk=' + str(top_k) + "_num-qrels=" + str(len(qrels))
    recall_path = 'bm25_saves/' + mode +  "_bm25recall@" + str(top_k) + "_num-qrels=" + str(len(qrels))

    if os.path.exists(idx_path) and os.path.exists(doc_path):

        with open(idx_path, "rb") as fp:
            doc_indeces = pickle.load(fp)

        with open(doc_path, "rb") as fp:
            doc_texts = pickle.load(fp)


    # Otherwise make new ones.
    else:
        doc_indeces, doc_texts, recall = make_bm25_docs(corpus, queries, qrels, top_k)

        with open(idx_path, "wb") as fp:
            pickle.dump(doc_indeces, fp)

        with open(doc_path, "wb") as fp:
            pickle.dump(doc_texts, fp)

        with open(recall_path, "w") as fp:
            fp.write("Recall@" + str(top_k) + " = " +  str(recall))

    return doc_indeces, doc_texts
