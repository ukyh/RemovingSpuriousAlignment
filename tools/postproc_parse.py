import json
import sys
import os
import itertools
import statistics
import random
from multiprocessing import Pool
from collections import Counter
from utils import trace


CORPUS = str(sys.argv[1])
DATA_PATH = str(sys.argv[2])
CORPUS_CONLLU = str(sys.argv[3])
MAX_SENT = int(sys.argv[4])
MIN_SENT_LEN = int(sys.argv[5])
MAX_FROM_OBJ = int(sys.argv[6])
WORKERS = int(sys.argv[7])
OID = str(sys.argv[8])
OBJ_PATH_JSON = os.path.join(DATA_PATH, "obj_path_{}_max{}.json".format(CORPUS, MAX_FROM_OBJ))
PAIR_PATH_JSON = os.path.join(DATA_PATH, "pair_path_{}_max{}.json".format(CORPUS, MAX_FROM_OBJ))
OBJ_FEAT_JSON = os.path.join(DATA_PATH, "obj_feat_{}_max{}.json".format(CORPUS, MAX_FROM_OBJ))
PAIR_FEAT_JSON = os.path.join(DATA_PATH, "pair_feat_{}_max{}.json".format(CORPUS, MAX_FROM_OBJ))
VAL_TEST_JSON = os.path.join(DATA_PATH, "val_test_dict.json")
TRAIN_OBJ_JSON = os.path.join(DATA_PATH, "img_obj_train.json")
VAL_OBJ_JSON = os.path.join(DATA_PATH, "img_obj_val.json")
TEST_OBJ_JSON = os.path.join(DATA_PATH, "img_obj_test.json")
PLURAL_JSON = os.path.join(DATA_PATH, "plural_words.json")
EVAL_JSON = os.path.join(DATA_PATH, "captions_val2014.json")
if OID == "v4":
    OBJ_PATH_JSON = OBJ_PATH_JSON.rstrip("json").rstrip(".") + "_v4.json"
    PAIR_PATH_JSON = PAIR_PATH_JSON.rstrip("json").rstrip(".") + "_v4.json"
    OBJ_FEAT_JSON = OBJ_FEAT_JSON.rstrip("json").rstrip(".") + "_v4.json"
    PAIR_FEAT_JSON = PAIR_FEAT_JSON.rstrip("json").rstrip(".") + "_v4.json"
    VAL_TEST_JSON = VAL_TEST_JSON.rstrip("json").rstrip(".") + "_v4.json"
    TRAIN_OBJ_JSON = TRAIN_OBJ_JSON.rstrip("json").rstrip(".") + "_v4.json"
    VAL_OBJ_JSON = VAL_OBJ_JSON.rstrip("json").rstrip(".") + "_v4.json"
    TEST_OBJ_JSON = TEST_OBJ_JSON.rstrip("json").rstrip(".") + "_v4.json"
    PLURAL_JSON = PLURAL_JSON.rstrip("json").rstrip(".") + "_v4.json"




if os.path.exists(VAL_TEST_JSON) \
    and os.path.exists(OBJ_PATH_JSON) and os.path.exists(OBJ_FEAT_JSON) \
    and os.path.exists(PAIR_PATH_JSON) and os.path.exists(PAIR_FEAT_JSON):
    trace("Files below already exist:")
    print(VAL_TEST_JSON)
    print(OBJ_PATH_JSON)
    print(OBJ_FEAT_JSON)
    print(PAIR_PATH_JSON)
    print(PAIR_FEAT_JSON)
    sys.exit(0)


#
# Load {obj, pair}-img dict:
#   {obj:["file1", ...], ...}
#   {pair:["file1", ...], ...}
#
obj_img_dict = {}
pair_img_dict = {}
with open(TRAIN_OBJ_JSON, encoding="utf-8") as f:
    indict = json.load(f)
    for fname, objs in indict.items():
        objs = list(set(objs))
        if len(objs) > 0:
            for obj in objs:
                if obj in obj_img_dict:
                    obj_img_dict[obj].append(fname)
                else:
                    obj_img_dict[obj] = [fname]
        if len(objs) > 1:
            _pairs = list(itertools.combinations(objs, 2))
            pairs = ["\t".join(sorted(pair_tup)) for pair_tup in _pairs]
            for pair in pairs:
                if pair in pair_img_dict:
                    pair_img_dict[pair].append(fname)
                else:
                    pair_img_dict[pair] = [fname]

trace("Total number of unique detected objects in train:", len(obj_img_dict))
trace("Total number of unique detected pairs in train:", len(pair_img_dict))


#
# Make val/test dict
#   { val:{ img:["cap1", ...] }
#     test:{ img:["cap1", ...] } }
#
val_test_dict = {"val":{}, "test":{}}
with open(VAL_OBJ_JSON, encoding="utf-8") as f:
    val_dict = json.load(f)
    val_test_dict["val"] = {k:[] for k in val_dict}
with open(TEST_OBJ_JSON, encoding="utf-8") as f:
    test_dict = json.load(f)
    val_test_dict["test"] = {k:[] for k in test_dict}
with open(EVAL_JSON, encoding="utf-8") as f:
    caption_data = json.load(f)
    id_to_name = [(x['id'], x['file_name']) for x in caption_data['images']]
    id_to_name = dict(id_to_name)
    for items in caption_data['annotations']:
        fname = id_to_name[items['image_id']]
        cap = items['caption']
        if fname in val_test_dict["val"]:
            val_test_dict["val"][fname].append(cap)
        elif fname in val_test_dict["test"]:
            val_test_dict["test"][fname].append(cap)
        else:
            # some eval captions are put in train in Karpathy split
            pass

if os.path.exists(VAL_TEST_JSON):
    trace("{} already exists".format(VAL_TEST_JSON))
else:
    with open(VAL_TEST_JSON, "w", encoding="utf-8") as outfile:
        json.dump(val_test_dict, outfile, indent=4)

    trace("Saved to {}".format(VAL_TEST_JSON))
    trace("Total number of val images:", len( val_test_dict["val"].keys() ))
    trace("Total number of test images:", len( val_test_dict["test"].keys() ))
    trace("Total number of val captinos:", sum( [len(val_test_dict["val"][i]) for i in val_test_dict["val"].keys()] ))
    trace("Total number of test captions:", sum( [len(val_test_dict["test"][i]) for i in val_test_dict["test"].keys()] ))


#
# Make plural dict:
#   {plural:category, ...}
#
with open(PLURAL_JSON, encoding="utf-8") as f:
    plural_data = json.load(f)
    plural_dict = {plural:single for single, plural in plural_data.items()}

trace("Loaded plural dict")


#
# Make CORPUS dicts:
#   {idx: {nouns}}
#   {idx: ["order word pos parent_order dep_rel", ...]}
#
corpus_obj_dict = {}
corpus_dep_dict = {}
with open(CORPUS_CONLLU, encoding="utf-8") as f:
    nns = []
    deps = []
    count = 0
    for line in f:
        items = line.rstrip().split("\t")
        if len(items) > 1:
            word = items[1].lower()             # make it lower to match with ctrl objects
            if items[4].startswith("NN"):       # choose nouns
                if word in plural_dict:         # if plural category name, add its single form
                    word = plural_dict[word]
                nns.append(word)
            deps.append("{}\t{}\t{}\t{}\t{}".format(items[0], items[1].lower(), items[4], items[6], items[7]))  # "order[\t]word[\t]pos[\t]parent_order[\t]dep_rel"
        else:
            nns = set(nns)
            # NOTE: exclude sentence with length < MIN_SENT_LEN and 0-nn,
            #       `len(deps) - 1` to exclude added period
            if len(deps) - 1 >= MIN_SENT_LEN and len(nns) > 0:
                corpus_obj_dict[count] = nns
                corpus_dep_dict[count] = deps
                count += 1
            nns = []
            deps = []

trace("Total number of {} captions with minimum length of {}: {}".format(CORPUS, MIN_SENT_LEN, count))


#
# Make detected obj dict
#   {obj: {idxs}}
#
def match_obj(obj):
    tmp_obj_dict = {obj:[]}
    # if obj is compound, extract sentneces which contain both elements
    if len(obj.split()) > 1:
        _obj1, _obj2 = obj.split()
        for idx, nns in corpus_obj_dict.items():
            if _obj1 in nns and _obj2 in nns:
                tmp_obj_dict[obj].append(idx)
    else:
        for idx, nns in corpus_obj_dict.items():
            if obj in nns:
                tmp_obj_dict[obj].append(idx)
    tmp_obj_dict[obj] = set(tmp_obj_dict[obj])  # {obj: {idxs}}
    tmp_obj_stat = len(tmp_obj_dict[obj])
    return tmp_obj_dict, tmp_obj_stat

p = Pool(WORKERS)
args = [obj for obj in obj_img_dict.keys()]
returned_list = p.map(match_obj, args)
matched_obj_dict = {}
matched_obj_stat = []
for d, stat in returned_list:
    matched_obj_dict.update(d)
    matched_obj_stat.append(stat)

trace("Max matched sentences per object: {}".format(max(matched_obj_stat)))
trace("Min matched sentences per object: {}".format(min(matched_obj_stat)))
trace("Mean matched sentences per object: {}".format(sum(matched_obj_stat) / len(matched_obj_stat)))


#
# Make detected pair dict
#   {"obj1[\t]obj2": {idxs}}
#
def match_pair(pair):
    obj1, obj2 = pair.split("\t")
    intersect = matched_obj_dict[obj1] & matched_obj_dict[obj2]
    tmp_pair_dict = {pair:intersect}    # {"obj1[\t]obj2": {idxs}}}
    tmp_pair_stat = len(intersect)
    return tmp_pair_dict, tmp_pair_stat

p = Pool(WORKERS)
args = [pair for pair in pair_img_dict.keys()]
returned_list = p.map(match_pair, args)
matched_pair_dict = {}
matched_pair_stat = []
for d, stat in returned_list:
    matched_pair_dict.update(d)
    matched_pair_stat.append(stat)

trace("Max matched sentences per pair: {}".format(max(matched_pair_stat)))
trace("Min matched sentences per pair: {}".format(min(matched_pair_stat)))
trace("Mean matched sentences per pair: {}".format(sum(matched_pair_stat) / len(matched_pair_stat)))


###########################################################
## PATH EXTRACTOR
###########################################################

def extract_attr(obj, matched_deps):
    # if compound, take HEAD in attr extraction
    if len(obj.split()) > 1:
        obj = obj.split()[-1]
    full_obj = set(obj.split())
    obj_orders = []
    for sent in matched_deps:
        for word in sent:
            items = word.rstrip().split("\t")
            word = items[1]
            if word in plural_dict:
                word = plural_dict[word]
            if obj == word and items[2].startswith("NN"):
                obj_order = items[0]
                obj_orders.append(obj_order)
                obj_order = 0
                break
    assert len(matched_deps) == len(obj_orders), "different length between matched_deps and obj_orders"
    attrs = []
    for sent, obj_order in zip(matched_deps, obj_orders):
        attr_idxs = [int(obj_order) - 1]
        ordered = []
        w2i = {}
        for word in sent:
            items = word.rstrip().split("\t")
            word = items[1]
            if word in plural_dict:
                word = plural_dict[word]
            if items[3] == obj_order and items[4] == "amod":
                attr_idxs.append(int(items[0]) - 1)
            ordered.append(items[1])
            if word in w2i:
                w2i[word].append(int(items[0]) - 1)
            else:
                w2i[word] = [int(items[0]) - 1]
        in_path_obj = [w2i[w] for w in full_obj]
        in_path_obj = set(sum(in_path_obj, []))   # {obj_idx, ...}
        attr_idxs_no_obj = list(set(attr_idxs) - in_path_obj)   # [path_idx, ...]
        if len(attr_idxs_no_obj) >= 1:
            dist = [[abs(_obj - _attr) for _attr in attr_idxs_no_obj] for _obj in in_path_obj]
            min_dist = min(sum(dist, []))
        else:
            min_dist = 1e+20
        if MAX_FROM_OBJ == 0 or (len(attr_idxs_no_obj) >= 1 and min_dist <= (MAX_FROM_OBJ // 2)):
            attr_idxs.sort()
            attr_idxs_no_obj.sort()
            attrs.append( [" ".join(ordered).rstrip(".").rstrip()] + [attr_idxs, attr_idxs_no_obj] ) # [cap, [idxs], [idxs_no_obj]]

        # NOTE: for objects, max_sent is multiplied to afford the more iteration than pairs
        if len(attrs) >= MAX_SENT * 5:
            return attrs
    return attrs


def extract_midpath(pair, matched_deps):
    obj1, obj2 = pair.split("\t")
    # if compound, take HEAD in middle path extraction
    if len(obj1.split()) > 1:
        obj1 = obj1.split()[-1]
    if len(obj2.split()) > 1:
        obj2 = obj2.split()[-1]
    full_obj = [obj.split() for obj in pair.split("\t")]
    full_obj = set(sum(full_obj, []))   # {obj, ...}
    obj1_order = 0
    obj2_order = 0
    target_orders = []
    for sent in matched_deps:
        for word in sent:
            items = word.rstrip().split("\t")
            word = items[1]
            if word in plural_dict:
                word = plural_dict[word]
            if obj1 == word and items[2].startswith("NN"):
                obj1_order = int(items[0])
            if obj2 == word and items[2].startswith("NN"):
                obj2_order = int(items[0])
        assert obj1_order > 0 and obj2_order > 0, "object pair does not match in matched deps"
        target_orders.append([obj1_order, obj2_order])
        obj1_order = 0
        obj2_order = 0
    assert len(matched_deps) == len(target_orders), "different length between matched_deps and target_orders"
    paths = []
    for sent, targ in zip(matched_deps, target_orders):
        targ1, targ2 = targ
        ordered = []
        w2i = {}
        dist = abs(targ1 - targ2) - 1
        if MAX_FROM_OBJ == 0 or (dist > 1 and dist <= MAX_FROM_OBJ):
            for word in sent:
                items = word.rstrip().split("\t")
                ordered.append(items[1])
                word = items[1]
                if word in plural_dict:
                    word = plural_dict[word]
                if word in w2i:
                    w2i[word].append(int(items[0]) - 1)
                else:
                    w2i[word] = [int(items[0]) - 1]
            sorted_targ = sorted(targ)
            path_idx = [i for i in range(sorted_targ[0] - 1,  sorted_targ[1])]
            in_path_obj = [w2i[w] for w in full_obj]
            in_path_obj = set(sum(in_path_obj, []))   # {obj_idx, ...}
            path_idx_no_obj = list(set(path_idx) - in_path_obj)   # [path_idx, ...]
            path_idx.sort()
            path_idx_no_obj.sort()
            paths.append( [" ".join(ordered).rstrip(".").rstrip()] + [path_idx, path_idx_no_obj] )  # [cap, [idxs], [idxs_no_obj]]
            
            if len(paths) >= MAX_SENT:
                return paths

    return paths

###########################################################


#
# Extract attribute path
#   {obj: [[text, [attr_indexes], [attr_indexes_no_obj]], ...]}
# NOTE: period at the end is removed
#       index starts from `0`
#
def get_attr(obj, matched_idxs):
    matched_idxs = list(matched_idxs)
    random.shuffle(matched_idxs)
    matched_deps = [corpus_dep_dict[idx] for idx in list(matched_idxs)] # [["order word pos parent_order dep_rel", ...], ...] 
    attr = extract_attr(obj, matched_deps)
    tmp_attr_dict = {obj:attr}  # {obj: [[text, [idxs], [idxs_no_obj]], ...]}
    return tmp_attr_dict

def wrapper_attr(args):
    return get_attr(*args)

if os.path.exists(OBJ_PATH_JSON):
    if os.path.exists(OBJ_FEAT_JSON):
        trace("{} already exists".format(OBJ_PATH_JSON))
    else:
        with open(OBJ_PATH_JSON, encoding="utf-8") as f:
            attr_dict = json.load(f)
        trace("Loaded {}".format(OBJ_PATH_JSON))
else:
    p = Pool(WORKERS)
    args = [(obj, matched_idxs) for obj, matched_idxs in matched_obj_dict.items()]
    returned_list = p.map(wrapper_attr, args)
    attr_dict = {}
    attr_stat = []
    for d in returned_list:
        if len(list(d.values())[0]) > 0:    # discard no-match dicts
            attr_dict.update(d)
            attr_stat.append(len(list(d.values())[0]))

    with open(OBJ_PATH_JSON, "w", encoding="utf-8") as outfile:
        json.dump(attr_dict, outfile, indent=4)
    
    trace("Saved to {}".format(OBJ_PATH_JSON))
    trace("Total of valid objects: {}".format(len(attr_stat)))
    trace("Max valid sentences per object: {}".format(max(attr_stat)))
    trace("Min valid sentences per object: {}".format(min(attr_stat)))
    trace("Mean valid sentences per object: {}".format(sum(attr_stat) / len(attr_stat)))
    trace("Median of valid sentences per object: {}".format(statistics.median(attr_stat)))
    trace("Mode of valid sentences per object: {}".format(Counter(attr_stat).most_common(3)))
    trace("Std of valid sentences per object: {}".format(statistics.stdev(attr_stat)))


#
# Make obj-img dict
#   {obj: [[file_names], ...]}
#
if os.path.exists(OBJ_FEAT_JSON):
    trace("{} already exists".format(OBJ_FEAT_JSON))
else:
    obj_feat_dict = {obj:obj_img_dict[obj] for obj in attr_dict}
    obj_feat_stat = [len(obj_img_dict[obj]) for obj in attr_dict]

    with open(OBJ_FEAT_JSON, "w", encoding="utf-8") as outfile:
        json.dump(obj_feat_dict, outfile, indent=4)
    
    trace("Saved to {}".format(OBJ_FEAT_JSON))
    trace("Max image per object: {}".format(max(obj_feat_stat)))
    trace("Min image per object: {}".format(min(obj_feat_stat)))
    trace("Mean image per object: {}".format(sum(obj_feat_stat) / len(obj_feat_stat)))
    trace("Median of image per object: {}".format(statistics.median(obj_feat_stat)))
    trace("Mode of image per object: {}".format(Counter(obj_feat_stat).most_common(3)))
    trace("Std of image per object: {}".format(statistics.stdev(obj_feat_stat)))


#
# Extract middle path
#   {"obj1[\t]obj2": [[text, [midpath_indexes], [midpath_indexes_no_obj]], ...]}
# NOTE: period at the end is removed
#       index starts from `0`
#
def get_midpath(pair, matched_idxs):
    matched_idxs = list(matched_idxs)
    random.shuffle(matched_idxs)
    matched_deps = [corpus_dep_dict[idx] for idx in list(matched_idxs)]
    midpath = extract_midpath(pair, matched_deps)
    tmp_midpath_dict = {pair:midpath}  # {"obj1[\t]obj2": [[text, [idxs], [idxs_no_obj]], ...]}
    return tmp_midpath_dict

def wrapper_midpath(args):
    return get_midpath(*args)

if os.path.exists(PAIR_PATH_JSON):
    if os.path.exists(PAIR_FEAT_JSON):
        trace("{} already exists".format(PAIR_PATH_JSON))
    else:
        with open(PAIR_PATH_JSON, encoding="utf-8") as f:
            midpath_dict = json.load(f)
        trace("Loaded {}".format(PAIR_PATH_JSON))
else:
    p = Pool(WORKERS)
    args = [(pair, matched_idxs) for pair, matched_idxs in matched_pair_dict.items()]
    returned_list = p.map(wrapper_midpath, args)
    midpath_dict = {}
    midpath_stat = []
    for d in returned_list:
        if len(list(d.values())[0]) > 0:    # discard no-match dict
            midpath_dict.update(d)
            midpath_stat.append(len(list(d.values())[0]))

    with open(PAIR_PATH_JSON, "w", encoding="utf-8") as outfile:
        json.dump(midpath_dict, outfile, indent=4)
    
    trace("Saved to {}".format(PAIR_PATH_JSON))
    trace("Total of valid pairs: {}".format(len(midpath_stat)))
    trace("Max valid sentences per pair: {}".format(max(midpath_stat)))
    trace("Min valid sentences per pair: {}".format(min(midpath_stat)))
    trace("Mean valid sentences per pair: {}".format(sum(midpath_stat) / len(midpath_stat)))
    trace("Median of valid sentences per pair: {}".format(statistics.median(midpath_stat)))
    trace("Mode of valid sentences per pair: {}".format(Counter(midpath_stat).most_common(3)))
    trace("Std of valid sentences per pair: {}".format(statistics.stdev(midpath_stat)))


#
# Make pair-img dict
#   {pair: [[file_names], ...]}
#
if os.path.exists(PAIR_FEAT_JSON):
    trace("{} already exists".format(PAIR_FEAT_JSON))
else:
    pair_feat_dict = {pair:pair_img_dict[pair] for pair in midpath_dict}
    pair_feat_stat = [len(pair_img_dict[pair]) for pair in midpath_dict]

    with open(PAIR_FEAT_JSON, "w", encoding="utf-8") as outfile:
        json.dump(pair_feat_dict, outfile, indent=4)
    
    trace("Saved to {}".format(PAIR_FEAT_JSON))
    trace("Max image per pair: {}".format(max(pair_feat_stat)))
    trace("Min image per pair: {}".format(min(pair_feat_stat)))
    trace("Mean image per pair: {}".format(sum(pair_feat_stat) / len(pair_feat_stat)))
    trace("Median of image per pair: {}".format(statistics.median(pair_feat_stat)))
    trace("Mode of image per pair: {}".format(Counter(pair_feat_stat).most_common(3)))
    trace("Std of image per pair: {}".format(statistics.stdev(pair_feat_stat)))


trace("Finished extracting attrs/midpaths in {}".format(CORPUS))
