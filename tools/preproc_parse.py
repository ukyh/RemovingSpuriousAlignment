import sys
import os
import pickle
import nltk
from utils import trace


GCC_FILE = str(sys.argv[1])
SS_FILE = str(sys.argv[2])
GCC_OUT = GCC_FILE.rstrip("tsv") + "proc"
SS_OUT = SS_FILE.rstrip("pkl") + "proc"

# PTB punctuations
# https://github.com/aimagelab/speaksee/blob/master/speaksee/evaluation/tokenizer.py
PUNCT = {"''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
        ".", "?", "!", ",", ":", "-", "--", "...", ";"}


if os.path.exists(GCC_OUT):
    trace("{} already exists".format(GCC_OUT))
else:
    with open(GCC_FILE, encoding="utf-8", errors="ignore") as infile:
        sents = [line.split("\t")[0] for line in infile]
    sents = set(sents)
    counter = 0
    with open(GCC_OUT, "w", encoding="utf-8", errors="ignore") as outfile:
        for cap in sents:
            cap_for_parse = " ".join(cap.split())
            if len(cap_for_parse.split()) > 0:
                if cap_for_parse[0].islower():
                    cap_for_parse = cap_for_parse[0].upper() + cap_for_parse[1:]
                if cap_for_parse[-1] not in PUNCT:
                    cap_for_parse = cap_for_parse + "."
                outfile.write(cap_for_parse + "\n")
                counter += 1
    trace("Total unique number of GCC captions:", counter)


if os.path.exists(SS_OUT):
    trace("{} already exists".format(SS_OUT))
else:
    # NOTE: SS_FILE stores sentences as a list of lists
    with open(SS_FILE, "rb") as infile:
        slist = pickle.load(infile, encoding="utf-8", errors="ignore")
    counter = 0
    with open(SS_OUT, "w", encoding="utf-8", errors="ignore") as outfile:
        for sent in slist:
            raw_sent = " ".join(sent[1:-1]).strip()    # exclude <S> and </S>
            split_sent = nltk.sent_tokenize(raw_sent)  # ["s1", ...]
            for cap in split_sent:
                cap_for_parse = " ".join(cap.split())
                if len(cap_for_parse.split()) > 0:
                    if cap_for_parse[0].islower():
                        cap_for_parse = cap_for_parse[0].upper() + cap_for_parse[1:]
                    if cap_for_parse[-1] not in PUNCT:
                        cap_for_parse = cap_for_parse + "."
                    outfile.write(cap_for_parse + "\n")
                    counter += 1
    trace("Total number of SS captions:", counter)

