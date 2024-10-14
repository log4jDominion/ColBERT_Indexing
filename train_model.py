import datetime
import sys

from nltk.tokenize import word_tokenize

from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig

training_set = []
collection = []
index_name = sys.argv[4]
experiment_name = sys.argv[3]
checkpoint = ''
labels = []


def get_first_n_tokens(text, n):
    tokens = word_tokenize(text)
    return ''.join(tokens[:n])


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
    results = searcher.search(query, k=1000)

    ranked_list = []

    for passage_id, passage_rank, passage_score in zip(*results):
        ranked_list.append(labels[passage_id])
        # print(f"\t{labels[passage_id]} \t\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")

    return ranked_list


def test_colbert(trainingSet):
    print(f"***********************Indexing starts at {datetime.datetime.now()}*************************")

    global training_set
    training_set = trainingSet

    create_dataset(trainingSet)

    global checkpoint
    checkpoint = 'colbert-ir/colbertv2.0'

    train_colbert_model(2, 300)

    print(f'***********************Indexing ends at {datetime.datetime.now()}*************************')
