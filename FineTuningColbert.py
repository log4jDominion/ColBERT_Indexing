from colbert import Indexer, Searcher, Trainer
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig

if __name__ == '__main__':
    # with Run().context(RunConfig(nranks=1, experiment="sushi_trainings")):
    #     config = ColBERTConfig(
    #         nbits=2,
    #         root="/sushi_trainings",
    #     )
    #     indexer = Indexer(checkpoint='colbert-ir/colbertv2.0', config=config)
    #     indexer.index(name="sushi.training.index", collection="complete_training_set.tsv", overwrite=True)

    with Run().context(RunConfig(nranks=1)):
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

    with Run().context(RunConfig(nranks=1, experiment="sushi_trainings")):
        config = ColBERTConfig(
            root="/sushi_trainings",
        )
        searcher = Searcher(index="sushi.training.index", config=config)
        queries = Queries("complete_queries_list.tsv")
        ranking = searcher.search_all(queries, k=10)
        ranking.save("complete.queries.ranking.tsv")