import re
import copy
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import ticker
import random
import os

TAGS = {'<GO>':0, '<EOS>':1, '<UNK>':2, '<PAD>':3}

def load_data(file_name):

	with open(file_name, 'r', encoding='utf-8') as f:
		corpus = f.read()

	return corpus


# def convert_to_ascii(sentence):

# 	sentence = unicodedata.normalize('NFD', sentence)
# 	return ''.join([char for char in sentence if unicodedata.category(char)!='Mn'])


def get_tags():
	return TAGS


def preprocess(sentence):

	# sentence = convert_to_ascii(sentence.lower().strip())

	# preserving the numbers and the punctuation
	sentence = sentence.lower().strip()
	sentence = re.sub(r'([.\-,;?!])', r' \1 ', sentence)
	sentence = re.sub(r'[^a-z\u00DF-\u00FF.\-,;?!]', r' ', sentence)
	tokenized_sent = sentence.split()
	return tokenized_sent



def get_word_mappings(vocab):

	word2idx = copy.copy(TAGS)


	for idx, word in enumerate(vocab.keys(), len(TAGS)):
		word2idx[word] = idx

	idx2word = {idx:word for word, idx in word2idx.items()}
	return word2idx, idx2word

def text_to_ids(proc_sents, word2idx):

	indexed_text = []
	for i,sentence in enumerate(proc_sents):
#		if len(sentence)!=0:
		indexed_text.append([word2idx[word] if word in word2idx else word2idx['<UNK>'] for word in sentence ])
#		else:
#			print(i)

	return indexed_text



def get_vocab(proc_sents, max_vocab_size, min_freq):

	word_counts = {}
	for sentence in proc_sents:
		for word in sentence:
			if word in word_counts:
				word_counts[word]+=1
			else:
				word_counts[word]=1

	word_counts = Counter(word_counts).most_common(max_vocab_size)
	vocab = {word:count for word, count in word_counts if count>=min_freq}
	return vocab

def get_rejected_indices(src_proc_sents, tgt_proc_sents, max_sentence_len=50, tgt_lang='de'):

	rejected_indices = []
	if tgt_lang=='de':

		for i, (src_sentence, tgt_sentence) in enumerate(zip(src_proc_sents, tgt_proc_sents)):

			if len(src_sentence)==0 or len(tgt_sentence)==0:
				rejected_indices += [i-1, i, i+1]

			elif len(src_sentence)>max_sentence_len or len(tgt_sentence)>max_sentence_len:
				rejected_indices.append(i)
	else:
		
		for i, (src_sentence, tgt_sentence) in enumerate(zip(src_proc_sents, tgt_proc_sents)):

			if len(src_sentence)==0 or len(tgt_sentence)==0 or len(src_sentence)>max_sentence_len or len(tgt_sentence)>max_sentence_len:
				rejected_indices.append(i)

	return rejected_indices




def load_europarl_data(source_path, target_path, train_frac=0.1, max_vocab_size=None, max_sentence_len=50, min_freq=2):

	src_text = load_data(source_path)
	src_sentences = src_text.split('\n')
	tgt_text = load_data(target_path)
	tgt_sentences = tgt_text.split('\n')

	print(len(src_sentences))
	upto = int(train_frac*len(src_sentences))
	
	src_sentences = src_sentences[:upto]
	tgt_sentences = tgt_sentences[:upto]

	src_proc_sents = [preprocess(sentence) for sentence in src_sentences]
	tgt_proc_sents = [preprocess(sentence) for sentence in tgt_sentences]

	rejected_indices = get_rejected_indices(src_proc_sents, tgt_proc_sents, max_sentence_len, 'de')

	src_proc_sents = [sentence for i, sentence in enumerate(src_proc_sents) if i not in rejected_indices]
	tgt_proc_sents = [sentence for i, sentence in enumerate(tgt_proc_sents) if i not in rejected_indices]

	src_vocab = get_vocab(src_proc_sents, max_vocab_size, min_freq)
	tgt_vocab = get_vocab(tgt_proc_sents, max_vocab_size, min_freq)

	src_w2i, src_i2w = get_word_mappings(src_vocab)
	tgt_w2i, tgt_i2w = get_word_mappings(tgt_vocab)

	src_indexed_text = text_to_ids(src_proc_sents, src_w2i)
	tgt_indexed_text = text_to_ids(tgt_proc_sents, tgt_w2i)

	return (src_indexed_text, src_w2i, src_i2w, tgt_indexed_text, tgt_w2i, tgt_i2w)

def load_cc_data(source_path, target_path, train_frac=0.1, max_vocab_size=None, max_sentence_len=50, min_freq=2):

	src_text = load_data(source_path)
	src_sentences = src_text.split('\n')
	tgt_text = load_data(target_path)
	tgt_sentences = tgt_text.split('\n')

	#print(len(src_sentences))
	num_samples = int(train_frac*len(src_sentences))
	
	rand_ints = random.sample(range(len(src_sentences)), num_samples)

	src_sents = [src_sentences[i] for i in rand_ints]
	tgt_sents = [tgt_sentences[i] for i in rand_ints]

	src_proc_sents = [preprocess(sentence) for sentence in src_sentences]
	tgt_proc_sents = [preprocess(sentence) for sentence in tgt_sentences]

	src_vocab = get_vocab(src_proc_sents, max_vocab_size, min_freq)
	tgt_vocab = get_vocab(tgt_proc_sents, max_vocab_size, min_freq)

	src_w2i, src_i2w = get_word_mappings(src_vocab)
	tgt_w2i, tgt_i2w = get_word_mappings(tgt_vocab)

	src_indexed_text = text_to_ids(src_proc_sents, src_w2i)
	tgt_indexed_text = text_to_ids(tgt_proc_sents, tgt_w2i)

	return (src_indexed_text, src_w2i, src_i2w, tgt_indexed_text, tgt_w2i, tgt_i2w)


def get_hind_en_sentences(lines):

	english_sentences, hindi_sentences = ([],[])
	for line in lines:
		cols = line.split('\t')
		if len(cols) < 5:
			continue
		if cols[2]=='manual' or cols[2]=='implied':
			english_sentences.append(cols[-2])
			hindi_sentences.append(cols[-1])

	return english_sentences, hindi_sentences

def preprocess_hindi(sentence):

	sentence = re.sub(r'([\u0964.\-,;?!])', r' \1 ', sentence)
	sentence = re.sub(r'[^\u0900-\u097F!.\-,;?]', r' ', sentence)
	tokenized_sent = sentence.split()
	return tokenized_sent


def load_hind_en_data(file_path, train_frac=1.0, max_vocab_size=None, max_sentence_len=50, min_freq=2):

	corpus = load_data(file_path)
	lines = corpus.split('\n')
	english_sentences, hindi_sentences = get_hind_en_sentences(lines)
	upto = int(train_frac*len(english_sentences))

	english_sentences = english_sentences[:upto]
	hindi_sentences = hindi_sentences[:upto]

	english_proc_sents = [preprocess(sentence) for sentence in english_sentences]
	hindi_proc_sents = [preprocess_hindi(sentence) for sentence in hindi_sentences]

	rejected_indices = get_rejected_indices(english_proc_sents, hindi_proc_sents, max_sentence_len, 'hi')
	english_proc_sents = [sentence for i, sentence in enumerate(english_proc_sents) if i not in rejected_indices]
	hindi_proc_sents = [sentence for i, sentence in enumerate(hindi_proc_sents) if i not in rejected_indices]

	english_vocab = get_vocab(english_proc_sents, max_vocab_size, min_freq)
	hindi_vocab = get_vocab(hindi_proc_sents, max_vocab_size, min_freq=0)

	eng_w2i, eng_i2w = get_word_mappings(english_vocab)
	hin_w2i, hin_i2w = get_word_mappings(hindi_vocab)

	eng_indexed_text = text_to_ids(english_proc_sents, eng_w2i)
	hin_indexed_text = text_to_ids(hindi_proc_sents, hin_w2i)

	return (eng_indexed_text, eng_w2i, eng_i2w, hin_indexed_text, hin_w2i, hin_i2w)

def load_eval_data(src_path, tgt_path, src_w2i, tgt_w2i, num_valid, max_sentence_len=50, tgt_lang='de', random_shuffle=False):
	
	src_corpus = load_data(src_path)
	tgt_corpus = load_data(tgt_path)

	if random_shuffle:
		src_sents = src_corpus.split('\n')
		tgt_sents = tgt_corpus.split('\n')
		if num_valid>len(src_sents):
			num_valid = len(src_sents)
		rand_ints = random.sample(range(len(src_sents)), num_valid)

		src_sents = [src_sents[i] for i in rand_ints]
		tgt_sents = [tgt_sents[i] for i in rand_ints]

	else:
		src_sents = src_corpus.split('\n')[:num_valid]
		tgt_sents = tgt_corpus.split('\n')[:num_valid]

	src_sents = [preprocess(sentence) for sentence in src_sents]

	if tgt_lang=='de':
		tgt_sents = [preprocess(sentence) for sentence in tgt_sents]
	else:
		tgt_sents = [preprocess_hindi(sentence) for sentence in tgt_sents]

	rejected_indices = get_rejected_indices(src_sents, tgt_sents, max_sentence_len, tgt_lang)

	src_sents = [sentence for i, sentence in enumerate(src_sents) if i not in rejected_indices]
	tgt_sents = [sentence for i, sentence in enumerate(tgt_sents) if i not in rejected_indices]

	src_indexed_text = text_to_ids(src_sents, src_w2i)
	tgt_indexed_text = text_to_ids(tgt_sents, tgt_w2i)

	return src_indexed_text, tgt_indexed_text


def plot_attention(attn_weights, source_sentence, output_sentence):
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(attn_weights, cmap='gray')
	src_labels = [''] + source_sentence
	out_labels = [''] + output_sentence

	ax.set_xticklabels(src_labels, rotation=90)
	ax.set_yticklabels(out_labels)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

	plt.show()


# def get_processed_data(file_path, train_frac=1.0, max_vocab_size=None, min_freq=2):


# 	text = load_data(file_path)
# 	sentences = text.split('\n')
# 	print(len(sentences))
# 	upto = int(train_frac*len(sentences))
# 	print(upto)
# 	sentences = sentences[:upto]
# 	proc_sents = [preprocess(sentence) for sentence in sentences]

# 	vocab = get_vocab(proc_sents, max_vocab_size, min_freq)
# 	word2idx, idx2word = get_word_mappings(vocab)
# 	data = text_to_ids(proc_sents, word2idx)

# 	return (data, word2idx, idx2word)	

def process_batch(src_batch, tgt_batch):

	batch_size = len(src_batch)
	src_sentence_lengths = sorted([len(sentence) for sentence in src_batch], reverse=True)
	src_batch, tgt_batch = (list(data_batch) for data_batch in zip(*sorted(zip(src_batch, tgt_batch), key=lambda t: -len(t[0]))))
	max_src_len = src_sentence_lengths[0]
	max_tgt_len = max([len(sentence) for sentence in tgt_batch])
	# print(max_tgt_len)
	proc_src_batch = [sentence + [TAGS['<EOS>']] + [TAGS['<PAD>']] * (max_src_len - len(sentence)) for sentence in src_batch]
	proc_tgt_batch = [[TAGS['<GO>']] + sentence + [TAGS['<EOS>']] + [TAGS['<PAD>']] * (max_tgt_len - len(sentence)) for sentence in tgt_batch]

	return proc_src_batch, proc_tgt_batch, src_sentence_lengths 


def prepend_go(data):

	return [[TAGS['<GO>']] + sentence for sentence in data]
# WIP
def generate_batches(source, target, batch_size=64):

	num_samples = len(source)

	for batch_i in range(0, num_samples//batch_size):
		start = batch_i*batch_size
		source_batch = source[start: start+batch_size]
		target_batch = target[start: start+batch_size]
		source_batch, target_batch, source_lengths = process_batch(source_batch, target_batch)
		yield source_batch, target_batch, source_lengths


def save_objects(obj, file_name):

	with open(file_name, 'wb') as f:
		pickle.dump(obj,f)

def load_objects(file_path, file_name):

	file = os.path.join(file_path, file_name)
	with open(file, 'rb') as f:
		obj = pickle.load(f)
	return obj