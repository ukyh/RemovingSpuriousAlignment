import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()

    # dataset
    parser.add_argument("--corpus", default="ss", choices=["ss", "gcc"], type=str)
    parser.add_argument("--max_pos_dist", default=4, type=int)
    parser.add_argument("--auto_setting", default=False, action="store_true")
    parser.add_argument("--img_pipe", default="inceptionv4", choices=["inceptionv4", "resnet101v2"], type=str)
    parser.add_argument("--oid", default="v2", choices=["v2", "v4"], type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--img_dir", default="~/mscoco_image_features", type=str)
    parser.add_argument("--val_test_path", default="val_test_dict.json", type=str)
    parser.add_argument("--vocab_path", default="word_counts.txt", type=str)
    parser.add_argument("--plural_path", default="plural_words.json", type=str)
    parser.add_argument("--train_obj_path", default="img_obj_train.json", type=str)
    parser.add_argument("--val_obj_path", default="img_obj_val.json", type=str)
    parser.add_argument("--test_obj_path", default="img_obj_test.json", type=str)

    # load
    parser.add_argument("--min_freq", default=5, type=int)
    parser.add_argument("--max_data", default=-1, type=int)

    # loop
    parser.add_argument("--epoch_size", default=100, type=int)
    parser.add_argument("--batch_train", default=8, type=int)
    parser.add_argument("--batch_eval", default=32, type=int)
    parser.add_argument("--early_stop", default=20, type=int)

    # cpu/gpu
    parser.add_argument("--device", default="cpu", type=str)

    # model
    parser.add_argument("--encoder", default="lstm", choices=["lstm", "gru"], type=str)
    parser.add_argument("--init_method", default="default", type=str)
    parser.add_argument("--out_pad", default=0., type=float)

    # train
    parser.add_argument("--use_gate", default=False, action="store_true")
    parser.add_argument("--use_pseudoL", default=False, action="store_true")
    parser.add_argument("--use_unique", default=False, action="store_true")
    parser.add_argument("--pos_gate_weight", default=1., type=float)
    parser.add_argument("--loss_weight", default=1., type=float)
    parser.add_argument("--norm_img", default=False, action="store_true")
    parser.add_argument("--accumulation_size", default=1, type=int)

    # eval
    parser.add_argument("--decode_size", default=20, type=int)
    parser.add_argument("--beam_size", default=1, type=int)
    parser.add_argument("--beam_len_norm", default=0., type=float)
    parser.add_argument("--beam_with_gate", default=0., type=float)

    # dimension
    parser.add_argument("--dim_tok", default=512, type=int)
    parser.add_argument("--dim_hid", default=512, type=int)
    parser.add_argument("--dim_img", default=1536, type=int)
    
    # dropout rate
    parser.add_argument("--dropout_seq", default=0.2, type=float)

    # n-layer
    parser.add_argument("--nlayer_enc", default=1, type=int)

    # optimizer
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--l2_decay", default=0., type=float)
    parser.add_argument('--grad_clip', default=0., type=float)

    # seed
    parser.add_argument("--seed", default=0, type=int)

    # save and load
    parser.add_argument("--save", default=False, action="store_true")
    parser.add_argument("--save_dir", default="./saved_models", type=str)
    parser.add_argument("--model_path", default="saved", type=str)
    parser.add_argument("--gen_path", default="output.json", type=str)
    parser.add_argument("--analysis", default=False, action="store_true")

    # log
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--log_path", default="", type=str)

    # selfcap option
    parser.add_argument('--min_intersect', default=2, type=int)

    args = parser.parse_args()

    # auto setting
    if args.auto_setting:
        if args.corpus == "ss":
            args.img_pipe = "inceptionv4"
            args.oid = "v2"
            args.encoder = "lstm"
            args.dim_hid = 512
            args.dim_tok = args.dim_hid
        if args.corpus == "gcc":
            args.img_pipe = "resnet101v2"
            args.oid = "v4"
            args.encoder = "gru"
            args.dim_hid = 200
            args.dim_tok = args.dim_hid
        if args.img_pipe == "inceptionv4":
            args.dim_img = 1536
        if args.img_pipe == "resnet101v2":
            args.dim_img = 2048
    
    # compose path
    args.attr_path = "obj_path_{}_max{}.json".format(args.corpus, args.max_pos_dist)
    args.attr_img_path = "obj_feat_{}_max{}.json".format(args.corpus, args.max_pos_dist)
    args.rel_path = "pair_path_{}_max{}.json".format(args.corpus, args.max_pos_dist)
    args.rel_img_path = "pair_feat_{}_max{}.json".format(args.corpus, args.max_pos_dist)
    args.train_feat_path = "img_{}_train.hdf5".format(args.img_pipe)
    args.val_feat_path = "img_{}_val.hdf5".format(args.img_pipe)
    args.test_feat_path = "img_{}_test.hdf5".format(args.img_pipe)
    args.model_path = os.path.join(args.save_dir, args.model_path)
    args.param_path = os.path.join(args.model_path, "model.pth")
    args.t2i_path = os.path.join(args.model_path, "t2i.json")
    args.gen_path = os.path.join(args.model_path, args.gen_path)
    if args.save and not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if args.log_path != "":
        args.log_path = os.path.join(args.log_dir, args.log_path)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

    # OID adjustment
    if args.oid == "v4":
        args.attr_path = args.attr_path.rstrip("json").rstrip(".") + "_v4.json"
        args.attr_img_path = args.attr_img_path.rstrip("json").rstrip(".") + "_v4.json"
        args.rel_path = args.rel_path.rstrip("json").rstrip(".") + "_v4.json"
        args.rel_img_path = args.rel_img_path.rstrip("json").rstrip(".") + "_v4.json"
        args.val_test_path = args.val_test_path.rstrip("json").rstrip(".") + "_v4.json"
        args.plural_path = args.plural_path.rstrip("json").rstrip(".") + "_v4.json"
        args.train_obj_path = args.train_obj_path.rstrip("json").rstrip(".") + "_v4.json"
        args.val_obj_path = args.val_obj_path.rstrip("json").rstrip(".") + "_v4.json"
        args.test_obj_path = args.test_obj_path.rstrip("json").rstrip(".") + "_v4.json"

    return args

