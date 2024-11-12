import datetime
import sys


import pandas as pd
import torch

from colbert import Indexer, Searcher, Trainer
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig

import faiss

training_set = []
collection = []
index_name = sys.argv[4]
experiment_name = sys.argv[3]
checkpoint = ''
labels = []

queries_ranked_list = []
index = 0
file_index = 0


def create_dataset(trainingSet):
    id = []
    docno = []
    folder = []
    box = []
    title = []
    ocr = []
    folder_label = []

    for idx, dictA in enumerate(trainingSet):
        id.append(idx)
        docno.append(dictA["docno"])
        folder.append(dictA["folder"])
        box.append(dictA["box"])
        title.append(dictA["title"])
        ocr.append(dictA["ocr"])
        folder_label.append(dictA["folderlabel"])


    df = pd.DataFrame(trainingSet)
    df.to_csv(f'complete_training_set.tsv', sep='\t', index=True)

    merged_text = []
    for m, n, o in zip(title, ocr, folder_label):
        if type(o) is tuple:
            o = ''.join(o)
        merged_text.append(m + ' ' + o + ' ' + n)

    print('Merged Text : ', merged_text[0])

    global collection
    collection = merged_text

    global labels
    labels = folder


def train_colbert_model(nbits, doc_maxlen):
    print(f"Indexing: {len(collection)} records")
    with Run().context(RunConfig(nranks=1, experiment='sushi')):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits,
                               kmeans_niters=4)  # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
        # Consider larger numbers for small datasets.

        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)

    indexer.get_index()


def colbert_search(query):
    with Run().context(RunConfig(experiment='sushi')):
        searcher = Searcher(index=index_name, collection=collection)

    # Find the top-3 passages for this query
    results = searcher.search(query, k=5)
    global index

    ranked_list = []

    for passage_id, passage_rank, passage_score in zip(*results):
        ranked_list.append(labels[passage_id])
        append_to_query_results({'id': index, 'passage_rank': passage_rank, 'passage_id': passage_id, 'passage_desc': labels[passage_id]})
        print({'id': index, 'passage_rank': passage_rank, 'passage_id': passage_id, 'passage_desc': labels[passage_id]})
        #print(f"\t{index} \t\t {labels[passage_id]} \t\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")

    index += 1
    return ranked_list


def init_query_ranked_list():
    global queries_ranked_list
    queries_ranked_list = []


def append_to_query_results(row):
    global queries_ranked_list
    queries_ranked_list.append(row)


def write_search_results():
    global file_index
    global queries_ranked_list
    df = pd.DataFrame(queries_ranked_list)

    file_name = f'queries_ranked_list_{file_index}.tsv'
    df.to_csv(file_name, sep='\t', header=False, index=True)
    file_index += 1



def test_colbert(trainingSet):
    print(f"***********************Indexing starts at {datetime.datetime.now()}*************************")

    init_query_ranked_list()

    global training_set
    training_set = trainingSet

    create_dataset(trainingSet)

    global checkpoint
    checkpoint = 'colbert-ir/colbertv2.0'

    train_colbert_model(2, 500)
    print(f'***********************Indexing ends at {datetime.datetime.now()}*************************')


