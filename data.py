from json import loads
import random
from sentence_transformers import InputExample

def check_query_size(query_id, queries, max_words, min_words):
    query = queries[query_id]
    N = len(query.split())

    if N > min_words and N < max_words:
        return query

    return None

def make_data_splits(
    path : str,
    train_size : int,
    eval_size : int,
    test_size : int,
    train_eval_size : int,
    splits : list = [0.8, 0.1, 0.1],
    min_words : int = 8,
    max_words : int = 256,
    seed : float = 32.) -> list:

    random.seed(seed)

    # Load in the data from json files.
    docs = {}
    with open(path + 'documents.json', 'r') as f:
        for line in f:
            line = dict(loads(line))
            docs[ line["id"] ] = line["title"] + '. ' + line["text"]
            # passages.append(line["title"] + '. ' +line['text'])

    queries = {}
    with open(path + 'queries.json', 'r') as f:
        for line in f:
            line = dict(loads(line))

            title = ""

            for w in line["title"].split():
                if not ('[' in w or ']' in w):
                    title += w + " "

            text = title + " " + line["description"]
            queries[ line["id"] ] = text

    # BM25 negatives
    with open(path + 'bm25_hard_negatives_all.json', 'r') as f:
        for line in f:
            bm25_negs = dict(loads(line)) # {"query id" : ["doc ids"] }

    # Training data.
    train_data = []  # [ (query, pos doc, [neg doc 1, neg doc 2, ..]), ... ]

    # Evaluation data.
    test_qrels_dict = {} # {"query id" : "doc id"}
    eval_qrels_dict = {} # {"query id" : "doc id"}
    train_qrels_dict = {} # {"query id" : "doc id"}

    # Shuffle the correct query-doc pairs before making training and eval splits.
    train_fract, test_fract, dev_fract = splits
    qrels = list(open(path + 'qrels.txt'))
    random.shuffle(qrels)

    train_split = int(train_fract * len(qrels))
    test_split = int(test_fract * len(qrels))
    dev_split = int(dev_fract * len(qrels))

    # Split the total dataset into a split of train, test and eval.
    train_qrels = qrels[ : train_split][ : train_size]
    test_qrels = qrels[train_split : train_split + test_split][ : test_size]
    eval_qrels = qrels[train_split + test_split : train_split + test_split + dev_split][ : eval_size]

    # Make the train data.
    for pair in train_qrels:
        query_id, _, doc_id, _ = pair.split()
        query = queries[query_id]
        doc = docs[doc_id]
        query = check_query_size(query_id, queries, max_words, min_words)

        if query:
            neg_docs = [ docs[doc_id] for doc_id in bm25_negs[query_id] if doc_id in docs ]

            if neg_docs: # make sure it's not empty
                train_data.append( InputExample(texts=[query, doc], label=1. ) )

                for ndoc in neg_docs:
                    # Multi-neg and cos-sim loss require different negative labels.
                    train_data.append( InputExample(texts=[query, ndoc], label=0.) )
  
                        

    # Make the eval data for testing on the training data.
    for pair in train_qrels[ : train_eval_size]:
        query_id, _, doc_id, _ = pair.split()
        if check_query_size(query_id, queries, max_words, min_words):
            train_qrels_dict[query_id] = doc_id
    
    # Make the evaluation data used for best model saving criterium.
    for pair in eval_qrels:
        query_id, _, doc_id, _ = pair.split()
        if check_query_size(query_id, queries, max_words, min_words):
            eval_qrels_dict[query_id] = doc_id

    # Make the testing evaluation data for when training is done.
    for pair in test_qrels:
        query_id, _, doc_id, _ = pair.split()
        if check_query_size(query_id, queries, max_words, min_words):
            test_qrels_dict[query_id] = doc_id


    return train_data, train_qrels_dict, eval_qrels_dict, test_qrels_dict, docs, queries

