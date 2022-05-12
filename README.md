
# Removing Word-Level Spurious Alignment between Images and Pseudo-Captions in Unsupervised Image Captioning

The code of our work, **Removing Word-Level Spurious Alignment between Images and Pseudo-Captions in Unsupervised Image Captioning** (EACL 2021). We proposed a simple and effective method to remove word-level spurious alignment between images and pseudo-captions for the better performance in unsupervised image captioning. Please refer to our paper for more details.  

[arXiv] https://arxiv.org/abs/2104.13872  
[ACL Anthology] https://www.aclweb.org/anthology/2021.eacl-main.323/


## Citation
```
@inproceedings{honda-etal-2021-removing,
    title = "Removing Word-Level Spurious Alignment between Images and Pseudo-Captions in Unsupervised Image Captioning",
    author = "Honda, Ukyo  and
      Ushiku, Yoshitaka  and
      Hashimoto, Atsushi  and
      Watanabe, Taro  and
      Matsumoto, Yuji",
    booktitle = "EACL",
    year = "2021",
}
```


## Requirements
```bash
git clone https://github.com/ukyh/RemovingSpuriousAlignment.git
cd RemovingSpuriousAlignment
pip install -r requirements.txt
```

If you would like to make the dataset by yourself (optional), install `nltk`, `spacy`, and `mosesdecoder` for text preprocessing.
```bash
pip install nltk==3.5
python -c "import nltk; nltk.download('punkt')"
pip install spacy==2.2.4
pip install spacy-conll==2.0.0
python -m spacy download en_core_web_lg

cd tools
git clone https://github.com/moses-smt/mosesdecoder
```


## Download Dataset
Download [full_data.tar.gz](https://drive.google.com/file/d/1e-JmBn62qVY9MDRx7niZS9Ta_g8kHC0h/view?usp=sharing) and unpack it in `data` directory.

**Acknowledgement**  
* `plural_words.json` and `word_counts.txt` are provided by [unsupervised_captioning](https://github.com/fengyang0317/unsupervised_captioning)
* `captions_train2014.json` and `captions_val2014.json` are provided by [MS COCO](https://cocodataset.org/)


## Make Dataset (Optional: The files can be downloaded as described above)
1. Download [Shutterstock corpus (sentences.pkl)](https://github.com/fengyang0317/unsupervised_captioning) and [Google's Conceptual Captions (Train-GCC-training.tsv)](https://ai.google.com/research/ConceptualCaptions/), and put them into `data` directory.

2. Follow the `Preprocess` instruction of [unsupervised_captioning_fast](https://github.com/ukyh/unsupervised_captioning_fast.git) to make the following items and copy them to `data` directory.
```
img_obj_test.json
img_obj_test_v4.json
img_obj_train.json
img_obj_train_v4.json
img_obj_val.json
img_obj_val_v4.json
```

3. Run the following commands.
```bash
# For Feng et al. (2019) setting
./get_data.sh --corpus ss --max_sent 400 --min_sent_len 5 --max_from_obj 4 --workers 70 --oid v2

# For Laina et al. (2019) setting
./get_data.sh --corpus gcc --max_sent 400 --min_sent_len 5 --max_from_obj 4 --workers 70 --oid v4
```


## Preprocess and Store Image Features
To prepare the features of MS COCO images, follow the `Preprocess` instruction of [unsupervised_captioning_fast](https://github.com/ukyh/unsupervised_captioning_fast.git). The image features will be saved to `~/mscoco_image_features` by default.


## Run
Commands to run the experiments.
```bash
# Run our full model in Feng et al. (2019) settings
python -u main.py --corpus ss --auto_setting --max_pos_dist 4 --max_data -1 --img_dir ~/mscoco_image_features --epoch_size 100 --batch_train 8 --batch_eval 32 --early_stop 20 --norm_img --use_gate --use_pseudoL --pos_gate_weight 16 --loss_weight 1 --use_unique --device 0

# Run our full model in Laina et al. (2019) settings
python -u main.py --corpus gcc --auto_setting --max_pos_dist 4 --max_data -1 --img_dir ~/mscoco_image_features --epoch_size 100 --batch_train 8 --batch_eval 32 --early_stop 20 --norm_img --use_gate --use_pseudoL --pos_gate_weight 16 --loss_weight 1 --use_unique --device 0

# Run w/o pseudoL model
python -u main.py --corpus ss --auto_setting --max_pos_dist 4 --max_data -1 --img_dir ~/mscoco_image_features --epoch_size 100 --batch_train 8 --batch_eval 32 --early_stop 20 --norm_img --use_gate --use_unique --device 0

# Run w/o gate model
python -u main.py --corpus ss --auto_setting --max_pos_dist 4 --max_data -1 --img_dir ~/mscoco_image_features --epoch_size 100 --batch_train 8 --batch_eval 32 --early_stop 20 --norm_img --use_unique --device 0

# Run w/o unique model
python -u main.py --corpus ss --auto_setting --max_pos_dist 4 --max_data -1 --img_dir ~/mscoco_image_features --epoch_size 100 --batch_train 8 --batch_eval 32 --early_stop 20 --norm_img --use_gate --use_pseudoL --pos_gate_weight 16 --loss_weight 1 --device 0

# Run w/o image model
python -u main_wo_img.py --corpus ss --auto_setting --max_pos_dist 4 --max_data -1 --img_dir ~/mscoco_image_features --epoch_size 100 --batch_train 8 --batch_eval 32 --early_stop 20 --norm_img --use_gate --use_pseudoL --pos_gate_weight 16 --loss_weight 1 --use_unique --device 0
```


## Preprocess to Combine
This is a preprocessing step to combine our method with [unsupervised_captioning_fast](https://github.com/ukyh/unsupervised_captioning_fast.git).

1. Train and save our full model. The command below will save the best model to `./saved_models/ss4_full`.
```bash
python -u main.py --corpus ss --auto_setting --max_pos_dist 4 --max_data -1 --img_dir ~/mscoco_image_features --epoch_size 100 --batch_train 8 --batch_eval 32 --early_stop 20 --norm_img --use_gate --use_pseudoL --pos_gate_weight 16 --loss_weight 1 --use_unique --device 0 --save --model_path ss4_full 
```

2. Load the saved model and generate captions for the training images. The command below will save the generated captions to `./saved_models/ss4_full/selfcap_ss4_mi2.json`.
```bash
python -u main_selfcap.py --corpus ss --auto_setting --max_pos_dist 4 --min_intersect 2 --img_dir ~/mscoco_image_features --batch_eval 32 --norm_img --use_gate --use_pseudoL --pos_gate_weight 16 --loss_weight 1 --use_unique --device 0 --model_path ss4_full --gen_path selfcap_ss4_mi2.json
```

3. Copy the generated caption file to `data` directory of [unsupervised_captioning_fast](https://github.com/ukyh/unsupervised_captioning_fast.git), then follow its `Combine` instruction.


## Notes
### Modified pseudo-caption preprocessing
We modified our pseudo-caption preprocessing to retain the sentences where `1 < n <= 4` words exist between a pair of detected objects, not `0 < n <= 4` as described in our EACL paper (description is corrected in the arXiv version).
We excluded the `n = 1` sentences as those sentences tended to ungramatically omit articles (_e.g._, **plant** on **table**). All results in our paper were obtained with the `1 < n <= 4` preprocessing.

### Incomplete seed fixing
We found that our seed fixing option could not completely control the learning of a model.
So seed fixing does not return exaclty the same results as shown in our paper, but we reran the experiments and confirmed that the results were almost the same as the ones in the paper on the average.


## References
* Yang Feng, Lin Ma, Wei Liu, and Jiebo Luo. 2019. Unsupervised image captioning. In _CVPR_.
* Iro Laina, Christian Rupprecht, and Nassir Navab. 2019. Towards unsupervised image captioning with shared multimodal embeddings. In _ICCV_.

