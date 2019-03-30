import helper
import argparse
import json
import os
import math
import pprint
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from model import Seq2Seq, Encoder, Decoder
from nltk.translate.bleu_score import corpus_bleu
#from nltk.translate.bleu_score import SmoothingFunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='NMT')
parser.add_argument('--mode', dest='mode', default='train', help='train or translate or calc_bleu', required=True)
parser.add_argument('--source_file', dest='source_file', default='data/train/training-parallel-europarl-v7/training/europarl-v7.de-en.en', help='location of the source file')
parser.add_argument('--target_file', dest='target_file', default='data/train/training-parallel-europarl-v7/training/europarl-v7.de-en.de', help='location of the target file')
parser.add_argument('--eval_source_file', dest='eval_source_file', default='data/dev/newstest2013.en', help='location of the eval source file')
parser.add_argument('--eval_target_file', dest='eval_target_file', default='data/dev/newstest2013.de', help='location of the eval target file')
parser.add_argument('--config_file', dest='config_file', default='config.json', help='config file name with path')
parser.add_argument('--save_path', dest='save_path', default='ckpt/', help='Path where model will be saved')
parser.add_argument('--load_path', dest='load_path', help='Path where model and vocab is present')
parser.add_argument('--sentence', nargs='+', help='the sentence to be translated')
parser.add_argument('-plot_attn', help='plot attention', action='store_true')
args = parser.parse_args()

if args.mode=='translate':
	if args.sentence==None:
		parser.error('provide the sentence to be translated using the --translate flag')
	if args.load_path==None:
		parser.error('provide the path where the model and vocab are present using --load_path flag')



def translate(model, sentences, lengths, tgt_i2w, max_len=50):
	
	model.eval()	

	tags = helper.get_tags()
	output, attn_weights = model.greedy_decode(sentences, lengths, tags['<GO>'], max_len)
	
	eos_idx = tags['<EOS>']
	trans_indexed_sents = []

	for i in range(output.shape[0]):
		eos_id_sent = np.where(output[i]==eos_idx)[0]
		if len(eos_id_sent)!=0 and eos_id_sent[0] > 0:
			trans_indexed_sents.append(output[i,:eos_id_sent[0]])
		else:
			trans_indexed_sents.append(output[i,:])

	translated_sents = [[tgt_i2w[word_id] for word_id in indexed_sent] for indexed_sent in trans_indexed_sents]

	return translated_sents, attn_weights


def calc_bleu(model, src, tgt, src_i2w, tgt_i2w, max_len=50):

	src, tgt, lengths = helper.process_batch(src, tgt)
	input_sents = torch.tensor(src, device=device)
	input_lengths = torch.tensor(lengths, device=device)
	
	trans_sents, _ = translate(model, input_sents, input_lengths, tgt_i2w, max_len)
	eos_idx = helper.get_tags()['<EOS>']
	tgt = [[tgt_i2w[word] for word in sent[1:sent.index(eos_idx)]] for sent in tgt]

	trans_sents = [sent for sent in trans_sents]
	ref_sents = [[sent] for sent in tgt]
	bleu_score = corpus_bleu(ref_sents, trans_sents)
	return bleu_score*100



def run_eval(model, eval_src, eval_tgt, target_vocab_size, pad_idx, batch_size):

	model.eval()
	avg_eval_loss = 0.0
	num_batches = len(eval_src)//batch_size
	for batch_i, (eval_src_batch, eval_tgt_batch, lengths) in enumerate(helper.generate_batches(eval_src, eval_tgt, batch_size)):


		eval_src_batch = torch.tensor(eval_src_batch, device=device)
		eval_tgt_batch = torch.tensor(eval_tgt_batch, device=device)
		lengths = torch.tensor(lengths, device=device)

		outputs = model(eval_src_batch, eval_tgt_batch, lengths)
		loss = F.nll_loss(outputs[:,1:,:].contiguous().view(-1, target_vocab_size), eval_tgt_batch[:,1:].contiguous().view(-1), ignore_index=pad_idx)
		avg_eval_loss += loss.item()

	avg_eval_loss/=num_batches

	return avg_eval_loss



def train(model, source, target, target_vocab_size, val_src, val_tgt, pad_idx, save_path, batch_size=64, epochs=10, learning_rate=0.01):
	
	train_loss = []
	val_loss = []
	perplexity = []
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	num_batches = len(source)//batch_size
	for epoch_i in range(epochs):	
		avg_train_loss = 0.0
		train_loss_per_200 = 0.0
		for batch_i, (source_batch, target_batch, lengths) in enumerate(helper.generate_batches(source, target, batch_size)):
			# print(source_batch.shape, target_batch.shape, lengths.shape)	
			model.train()
			source_batch = torch.tensor(source_batch, device=device)
			target_batch = torch.tensor(target_batch, device=device)
			lengths = torch.tensor(lengths, device=device)
			optimizer.zero_grad()
			outputs = model(source_batch, target_batch, lengths)
			loss = F.nll_loss(outputs[:,1:,:].contiguous().view(-1, target_vocab_size), target_batch[:,1:].contiguous().view(-1), ignore_index=pad_idx)
			# print(loss)
			loss.backward()
			clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			avg_train_loss += loss.item()
			train_loss_per_200 += loss.item()

			if batch_i%50==0 and batch_i!=0:
				avg_train_loss /= 50
				print('Epoch {:>3} Batch {:>4}/{} Avg Train Loss ({} batches):{:.6f}'.format(epoch_i,batch_i,num_batches, 50, avg_train_loss))
				avg_train_loss = 0.0

			if batch_i%200==0 and batch_i!=0:
				train_loss_per_200 /= 200
				avg_val_loss = run_eval(model, val_src, val_tgt, target_vocab_size, pad_idx, batch_size)
				print('Epoch {:>3} Batch {:>4}/{} Avg Val Loss ({} batches):{:.6f}'.format(epoch_i,batch_i,num_batches, 200, avg_val_loss))
				print('Avg Val Perplexity: {:.2f}'.format(math.exp(avg_val_loss)))
				train_loss.append(train_loss_per_200)
				val_loss.append(avg_val_loss)
				perplexity.append(math.exp(avg_val_loss))
				train_loss_per_200 = 0.0

	helper.save_objects((train_loss,val_loss), os.path.join(save_path, 'loss.pkl'))
	helper.save_objects(perplexity, os.path.join(save_path, 'perplexity.pkl'))
	print('------Training Complete------')
	file = os.path.join(save_path, 'model.pt')	
	torch.save(model.state_dict(), file)
	print('Model saved')



def build_model(config, input_vocab_size, target_vocab_size):

	embed_size = config['embed_size']
	hidden_size = config['hidden_size']
	proj_size = config['proj_size']
	enc_num_layers = config['enc_num_layers']
	dec_num_layers = config['dec_num_layers']
	dropout = config['dropout']
	attn_type = config['attn_type']
	self_attn = config['self_attn']
	intra_temp_attn = config['intra_temp_attn']
	dec_attn = config['dec_attn']
	if self_attn or intra_temp_attn:
		dec_attn = True


	encoder = Encoder(input_vocab_size, embed_size, hidden_size, enc_num_layers, dropout)
	decoder = Decoder(target_vocab_size, embed_size, hidden_size, dec_num_layers, proj_size, dropout, attn_type=attn_type, self_attn=self_attn, dec_attn=dec_attn, intra_temp_attn=intra_temp_attn)
	model = Seq2Seq(encoder, decoder).to(device)

	return model


def main():

	mode = args.mode
	source_file = args.source_file
	target_file = args.target_file
	eval_source_file = args.eval_source_file
	eval_target_file = args.eval_target_file
	save_path = args.save_path
	usr_sentence = args.sentence
	load_path = args.load_path
	plot_attn = args.plot_attn

	with open(args.config_file, 'r') as f:
		config = json.load(f)

	pprint.pprint(config)
	assert torch.cuda.is_available()

	if mode == 'train':
		if config['target_lang'] == 'de':
			source_data, source_word2idx, source_idx2word, target_data, target_word2idx, target_idx2word = helper.load_europarl_data(source_file, target_file, max_sentence_len=config['max_sentence_len'], train_frac=config['train_frac'], max_vocab_size=config['vocab_size'])
			# source_data, source_word2idx, source_idx2word, target_data, target_word2idx, target_idx2word = helper.load_cc_data(source_file, target_file, max_sentence_len=config['max_sentence_len'], train_frac=config['train_frac'], max_vocab_size=config['vocab_size'])

		elif config['target_lang'] == 'hi':
			source_data, source_word2idx, source_idx2word, target_data, target_word2idx, target_idx2word = helper.load_hind_en_data(source_file, train_frac=config['train_frac'], max_sentence_len=config['max_sentence_len'], max_vocab_size=config['vocab_size'])
		print(len(source_data))
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		helper.save_objects((source_word2idx, target_word2idx), os.path.join(save_path, 'word2idx.pkl'))
		helper.save_objects((source_idx2word, target_idx2word), os.path.join(save_path, 'idx2word.pkl'))
		model = build_model(config, len(source_word2idx), len(target_word2idx))
		pad_idx = helper.get_tags()['<PAD>'] 
		num_valid = int(0.2*len(source_data))
		val_src, val_tgt = helper.load_eval_data(eval_source_file, eval_target_file, source_word2idx, target_word2idx, num_valid, max_sentence_len=config['max_sentence_len'], tgt_lang=config['target_lang'])
		assert(len(val_src) == len(val_tgt))
		train(model, source_data, target_data, len(target_word2idx), val_src, val_tgt, pad_idx, save_path, config['batch_size'], config['epochs'], config['learning_rate'])

	elif mode == 'translate':
		sentence = ' '.join(usr_sentence)
		src_w2i, _ = helper.load_objects(load_path, 'word2idx.pkl')
		src_i2w, tgt_i2w = helper.load_objects(load_path, 'idx2word.pkl')
		model = build_model(config, len(src_w2i), len(tgt_i2w))
		model.load_state_dict(torch.load(os.path.join(load_path,'model.pt')))
		src_sent = helper.preprocess(sentence)
		src_length = [len(src_sent)]
		src_sent = [src_sent]
		# print(len(src_sent))
		
		src_sent = helper.text_to_ids(src_sent, src_w2i)	
		# print(src_length)
		input_sent = torch.tensor(src_sent, device=device)
		length = torch.tensor(src_length, device=device)
		translated_sents, attn_weights = translate(model, input_sent, length, tgt_i2w, max_len = config['max_sentence_len'])
		trans_sent = ' '.join(translated_sents[0])
		print(trans_sent)
		
		if plot_attn:
			src_sent = [[src_i2w[word] for word in sent] for sent in src_sent ]
			#print(attn_weights[0].shape)
			helper.plot_attention(attn_weights[0], src_sent[0], translated_sents[0])

	elif mode == 'calc_bleu':

		src_w2i, tgt_w2i = helper.load_objects(load_path, 'word2idx.pkl')
		src_i2w, tgt_i2w = helper.load_objects(load_path, 'idx2word.pkl')
		model = build_model(config, len(src_w2i), len(tgt_w2i))
		model.load_state_dict(torch.load(os.path.join(load_path,'model.pt')))
		src, tgt = helper.load_eval_data(eval_source_file, eval_target_file, src_w2i, tgt_w2i,6005, max_sentence_len=config['max_sentence_len'], tgt_lang=config['target_lang'], random_shuffle=False)
		bleu_score = calc_bleu(model, src, tgt, src_i2w, tgt_i2w, max_len=config['max_sentence_len'])
		print('Bleu score: {:.2f}'.format(bleu_score))
	
	else:
		parser.error('Not a valid mode try with train / translate / calc_bleu') 






	

if __name__ == '__main__':
	main()