# E1-246-Assignment2

## About
This repository contains the implementation of seq2seq model with attention mechanism for neural machine translation task. The code provides option to train the model with four different types of attention mechanisms:
1. Additive attention [1]
2. Multiplicative attention [2]
3. Scaled dot-procuct attention [3]
4. Key-Value attention [4]

## Setup Instructions

### Setting up the environment
Python 3.6 is required to run the code in this repository. I have used python 3.6.7 

To install the requirements
```
pip3 install -r requirements.txt
```

### Dataset


### Setting the hyperparameters
All the hyperparameters are loaded from `config.json` file. Here I have briefly described each of these hyperparameter flags present in `config.json`.
* `learning_rate` : learning rate required to train the model
* `batch_size` : training batch size
* `epochs` : number of epochs to train
* `embed_size` : the size of word embedding
* `vocab_size` : size of the vocabulary for source and target languages
* `hidden_size` : dimension of the hidden unit activations
* `enc_num_layers` : number of LSTM layers in the encoder
* `dec_num_layers` : number of LSTM layers in the decoder
* `proj_size` : projection vector size (required for Additive and Key-Value attention)
* `dropout` : dropout probability
* `attn_type` : the type of attention you want to use (takes values "additive", "multiplicative", "scaled dot-product", "key-value")
* `self_attn` : use self-attention true/false
* `target_lang` : target language for translation (takes values "de", "hi" for german and hindi respectively)
* `max_sentence_len` : maximum sentence length to be used
* `train_frac` : fraction of training data to be used for training (should be 1 if you wish to train on complete dataset)

### How to run

All the below commands assume that `config.json` is present in the same directory as the code. If you wish to load `config.json` from some other directory then please specify that using `--config_file` flag as `--config_file dirname/config.json` in all of the commands below

To train your model run:
```
python nmt.py --mode train --source_file data/train/europarl-v7.de-en.en --target_file data/train/europarl-v7.de-en.de --eval_source_file data/dev/newstest2013.en --eval_target_file data/dev/newstest2013.de --save_path ckpt/model_add_wsa_b128_ep10/  
```
To translate a sentence run:
```
python nmt.py --mode translate --sentence Madam President, I would like to thank Mr Poettering for advertising this debate --load_path ckpt/model_add_wsa_b128_ep10/ 
```
**Note**: Use the `-plot_attn` flag to plot the attention for the input sentence, by default this is false.

To get the bleu score for a evaluation dataset run:
```
python nmt.py --mode calc_bleu --eval_source_file data/dev/newstest2013.en --eval_target_file data/dev/newstest2013.de  --load_path ckpt/model_add_wsa_b128_ep10/
```

## Pretrained Model
The `ckpt` directory contains the trained models which can be used to reproduce some of the results in the report. **Note**: By default the config file in the code directory corresponds to `model_add_wsa_b128_ep10` if you wish to evaluate for some other task please use the specific config files in the respective model directories.

## References
<cite>[1] Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).</cite> <br>
<cite>[2]  Luong, Thang, Hieu Pham, and Christopher D. Manning. "Effective Approaches to Attention-based Neural Machine  Translation." Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing. 2015.</cite>
<cite>[3] Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.</cite>
<cite>[4] Liu, Yang, and Mirella Lapata. "Learning structured text representations." Transactions of the Association of Computational Linguistics 6 (2018): 63-75.</cite>
