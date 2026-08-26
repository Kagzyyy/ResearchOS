from beir import util
from beir.datasets.data_loader import GenericDataLoader

SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)


def main() -> None:
    data_path = util.download_and_unzip(SCIFACT_URL, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    print("Documents:", len(corpus))
    print("queries:", len(queries))
    print("qrels:", len(qrels))
    qid = list(queries.keys())[0]
    print("Claim: ", queries[qid])
    print("qrels entry:", qrels[qid])


if __name__ == "__main__":
    main()
