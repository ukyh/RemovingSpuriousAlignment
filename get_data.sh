#!/bin/bash

# Based on: https://github.com/facebookresearch/XLM/blob/master/get-data-nmt.sh

# Sample command
#   ./get_data.sh --corpus ss --max_sent 400 --min_sent_len 5 --max_from_obj 4 --workers 70 --oid v2
# NOTE:
#   `max_from_obj 0` means no filter


#
# Read arguments
#
set -e
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --corpus)
    CORPUS="$2"; shift 2;;
  --max_sent)
    MAX_SENT="$2"; shift 2;;
  --min_sent_len)
    MIN_SENT_LEN="$2"; shift 2;;
  --max_from_obj)
    MAX_FROM_OBJ="$2"; shift 2;;
  --workers)
    WORKERS="$2"; shift 2;;
  --oid)
    OID="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"
if [ "$CORPUS" == "" ]; then echo "--corpus {'gcc', 'ss'} not provided"; exit; fi
if [ "$CORPUS" != "gcc" ] && [ "$CORPUS" != "ss" ] && [ "$CORPUS" != "coco" ]; then echo "--corpus must be 'gcc', 'ss' or 'coco'"; exit; fi
if [ "$MAX_SENT" == "" ]; then echo "--the number of max sentences per pair not provided"; exit; fi
if [ "$MIN_SENT_LEN" == "" ]; then echo "--min sentence length not provided"; exit; fi
if [ "$MAX_FROM_OBJ" == "" ]; then echo "--max number of words from objects not provided"; exit; fi
if [ "$WORKERS" == "" ]; then echo "--the number of workers not provided"; exit; fi
if [ "$OID" != "v2" ] && [ "$OID" != "v4" ]; then echo "--oid must be 'v2' or 'v4'"; exit; fi


#
# Main path
#
DATA_PATH=$PWD/data
TOOL_PATH=$PWD/tools
GCC=$DATA_PATH/Train-GCC-training
SS=$DATA_PATH/sentence

if [ "$CORPUS" == "ss" ]; then
  TRG_CORPUS=$SS
elif [ "$CORPUS" == "gcc" ]; then
  TRG_CORPUS=$GCC
fi

MOSES=$TOOL_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l en -penn 0 | $REM_NON_PRINT_CHAR"


#
# Preprocess text
#
if ! [[ -f "$GCC.proc" ]] || ! [[ -f "$SS.proc" ]]; then
  echo "Preprocessing files for parsing"
  python -u $TOOL_PATH/preproc_parse.py $GCC.tsv $SS.pkl
fi


#
# Normalize strings
#
if ! [[ -f "$GCC.norm" ]]; then
  echo "Normalizing $GCC"
  eval "cat $GCC.proc | $PREPROCESSING > $GCC.norm"
fi
if ! [[ -f "$SS.norm" ]]; then
  echo "Normalizing $SS"
  eval "cat $SS.proc | $PREPROCESSING > $SS.norm"
fi


#
# Parse sentences
#
if ! [[ -f "$TRG_CORPUS.conllu" ]]; then
  python -u $TOOL_PATH/run_spacy.py $TRG_CORPUS.norm 0 $WORKERS
fi


#
# Extract pseudo-captions
#
python -u $TOOL_PATH/postproc_parse.py $CORPUS $DATA_PATH $TRG_CORPUS.conllu $MAX_SENT $MIN_SENT_LEN $MAX_FROM_OBJ $WORKERS $OID


echo `date` "Finished creating datasets"

