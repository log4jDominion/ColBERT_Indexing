import csv

from colbert import Indexer, Searcher, Trainer
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig

result_file_index = -1
labels = []


def fine_tuning_model(training_dataset):
    global result_file_index
    result_file_index += 1

    global labels
    labels = [dictionary['folder'] for dictionary in training_dataset]

    # with Run().context(RunConfig(nranks=1, experiment="sushi_trainings")):
    #     config = ColBERTConfig(
    #         nbits=2,
    #         root="/sushi_trainings",
    #     )
    #     indexer = Indexer(checkpoint='colbert-ir/colbertv2.0', config=config)
    #     indexer.index(name="sushi.training.index", collection="complete_training_set.tsv", overwrite=True)

    with Run().context(RunConfig(nranks=1, experiment="sushi_trainings")):
        triples = 'complete_triples.jsonl'
        queries = 'complete_queries_list.tsv'
        collection = 'complete_training_set.tsv'

        config = ColBERTConfig(
            bsize=32,
            root='/sushi_trainings')

        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)

        checkpoint_path = trainer.train()

        print(f"Saved checkpoint to {checkpoint_path}...")

        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name="sushi.training.index", collection="complete_training_set.tsv", overwrite=True)

    # with Run().context(RunConfig(nranks=1, experiment="sushi_trainings")):
    #     config = ColBERTConfig(
    #         root="/sushi_trainings",
    #     )
    #     searcher = Searcher(index="sushi.training.index", config=config)
    #     queries = Queries("complete_queries_list.tsv")
    #     ranking = searcher.search_all(queries, k=1000)
    #     return ranking.save(f"sushi.test.run.{result_file_index}.ranking.tsv")


def fetch_results(query):
    with Run().context(RunConfig(nranks=1, experiment="sushi_trainings")):
        config = ColBERTConfig(
            root="/sushi_trainings",
        )
        searcher = Searcher(index="sushi.training.index", config=config)

        results = searcher.search(query, k=1000)

        ranked_list = []

        for passage_id, passage_rank, passage_score in zip(*results):
            ranked_list.append(labels[passage_id])
            # print(f"\t{labels[passage_id]} \t\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")

        return ranked_list


# def fetch_results(query_index, rank_file_path):
#     result = []
#     with open(rank_file_path, mode='r', newline='') as file:
#         reader = csv.reader(file, delimiter='\t')
#
#         for row in reader:
#             if len(row) >= 2:  # Ensure there are at least two columns
#                 first_col_value = int(row[0])
#                 second_col_value = int(row[1])
#
#                 # Add the second column value to the list of the first column's key
#                 if first_col_value == query_index:
#                     result.append(labels[second_col_value])
#
#     return result


if __name__ == '__main__':
    fine_tuning_model()
