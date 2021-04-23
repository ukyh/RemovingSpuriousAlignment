import sys
import spacy
from spacy_conll import ConllFormatter
from utils import trace


IN_CORPUS = str(sys.argv[1])
KEEP_BOUND = int(sys.argv[2])
WORKERS = int(sys.argv[3])
OUT_CONLLU = ".".join(IN_CORPUS.split(".")[:-1] + ["conllu"])


model="en_core_web_lg"
nlp = spacy.load(model)
conllformatter = ConllFormatter(nlp)
nlp.add_pipe(conllformatter, after='parser')


# keep the passed boundaries of doc
# see below in `Sentence Segmentation`:
#   https://spacy.io/usage/linguistic-features
def disable_auto_boundaries(doc):
    for token in doc[1:]:
        doc[token.i].is_sent_start = False

    return doc

if KEEP_BOUND:
    trace("Disabled auto sentence segmentation")
    nlp.add_pipe(disable_auto_boundaries, before="parser")


with open(IN_CORPUS, encoding="utf-8", errors="ignore") as f:
    texts = [" ".join(text.split()) for text in f.readlines()]  # make sure to remove empty words


BATCH = len(texts) // WORKERS


trace("Parsing {} with {}".format(IN_CORPUS, model))
with open(OUT_CONLLU, "w", encoding="utf-8") as f:
    for doc in nlp.pipe(texts, n_process=WORKERS, batch_size=BATCH):
        f.write(doc._.conll_str + "\n")


trace("Finished parsing {} with {}".format(IN_CORPUS, model))
