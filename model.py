import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import attention as attn
import random
import math
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module): # OK tested

	# input_size : number of words in the input vocab
	# embed_size : embedding dimensions
	# hidden_size : number of hidden units
	# num_layers : number of lstm layers to be stacked
	# dropout : drop prob

	 def __init__(self, input_size, embed_size, hidden_size, num_layers, dropout_prob=0.2):

	 	super(Encoder, self).__init__()
	 	self.input_size = input_size
	 	self.embed_size = embed_size
	 	self.hidden_size = hidden_size
	 	self.num_layers = num_layers
	 	self.dropout_prob = dropout_prob
	 	self.embedding = nn.Embedding(input_size, embed_size)
	 	self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=True)


	 def forward(self, input_, lengths): 
	 	
	 	embed = self.embedding(input_)
	 	# print('inside encoder', embed.shape)
	 	# mask the paddings: expects inputs sorted in decresing order of their seq lengths
	 	packed = pack_padded_sequence(embed, lengths, batch_first=True)
	 	output, (final_hidden, _) = self.lstm(packed)

	 	# restore to original
	 	output, _ = pad_packed_sequence(output, batch_first=True) # [BxTx2H]

	 	# combine the final layer output of the bilstms
	 	#output = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]

	 	fwd_bkwd_split = final_hidden.size(0)//2
	 	fwd_hidden = final_hidden[:fwd_bkwd_split] # [num_layersxBxH]
	 	bkwd_hidden = final_hidden[fwd_bkwd_split:] # [num_layersxBxH]

	 	# concatenate the fwd and bkwd hidden layers
	 	fwd_bkwd_cat = torch.cat([fwd_hidden, bkwd_hidden], dim=2) #  [num_layersxBx2H]

	 	return output, fwd_bkwd_cat

class Decoder(nn.Module): # OK tested

	# output_size : number of words in the output vocabK
	# embed_size : embedding dimension
	# hidden_size : number of hidden units
	# num_layers : number of lstm layers to be stacked
	# dropout : drop prob

	def __init__(self, output_size, embed_size, hidden_size, num_layers, proj_size=128, dropout_prob=0.2, init_hidden_from_enc=True, attn_type='additive', self_attn=False, dec_attn=False,  intra_temp_attn=False):

		super(Decoder, self).__init__()
		self.output_size = output_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = num_layers
		self.dropout_prob = dropout_prob
		self.self_attn = self_attn
		self.dec_attn = dec_attn
		self.attn_type = attn_type
		self.intra_temp_attn = intra_temp_attn
		self.dropout = nn.Dropout(dropout_prob)
		self.embedding = nn.Embedding(output_size, embed_size)
		# attention on encoder states
		# self.enc_attention = Attention(hidden_size, 2*hidden_size, 2*hidden_size, proj_size, attn_type)
		# attention on decoder states if self_attn is true
		# self.dec_attention = Attention(hidden_size, hidden_size, hidden_size, proj_size, attn_type) if self_attn else None

		self.enc_self_attention = attn.SelfAttention(2*hidden_size, 2*hidden_size, 2*hidden_size, proj_size) if self_attn else None
		self.dec_self_attention = attn.SelfAttention(hidden_size, hidden_size, hidden_size, proj_size) if self_attn else None

		if attn_type=='multiplicative':
			self.enc_attention = attn.MultiplicativeAttention(hidden_size, 2*hidden_size)
			self.dec_attention = attn.MultiplicativeAttention(hidden_size, hidden_size) if dec_attn else None
			self.context_size = 3*hidden_size if dec_attn else 2*hidden_size
		elif attn_type=='scaled dot-product':
			self.enc_attention = attn.ScaledDotProductAttention(hidden_size, 2*hidden_size)
			self.dec_attention = attn.ScaledDotProductAttention(hidden_size, hidden_size) if dec_attn else None
			self.context_size = 3*hidden_size if dec_attn else 2*hidden_size
		elif attn_type=='key-value':
			self.enc_attention = attn.KeyValueAttention(hidden_size, 2*hidden_size, 2*hidden_size, hidden_size)
			self.dec_attention = attn.KeyValueAttention(hidden_size, hidden_size, hidden_size, hidden_size//2) if dec_attn else None
			self.context_size = 3*hidden_size//2 if dec_attn else hidden_size
		else:
			# default case additive
			self.enc_attention = attn.AdditiveAttention(hidden_size, 2*hidden_size, proj_size)
			self.dec_attention = attn.AdditiveAttention(hidden_size, hidden_size, proj_size) if dec_attn else None
			self.context_size = 3*hidden_size if dec_attn else 2*hidden_size

		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

		# input is [dec_hidden, context], in case of self attention context calculalted using [hf,hb,sf]
		self.out = nn.Linear(hidden_size + self.context_size, output_size)
		# to map encoder hidden state which is bidirectional to decoder unidirectional hidden state
		self.enc2dec_hidden = nn.Linear(2*hidden_size, hidden_size) if init_hidden_from_enc else None


	def forward(self, input_, enc_outputs, dec_last_hidden, dec_last_cell, dec_prev_outputs=None, enc_prev_attn_scores=None):

		embed = self.embedding(input_) # embedding of the current word
		embed = self.dropout(embed) # hack to prevent overfitting

		# calculate the attn weights and the context vector using dec_last_hidden state and enc_outputs (output layer hidden units of encoder)
		# consider the last layer hidden activation for the query vector


		# concat the embed and context_vec to prepare the dec lstm input
		lstm_output, (final_hidden, final_cell) = self.lstm(embed, (dec_last_hidden, dec_last_cell))

		# if self attention is true then concatenate the context calculated from decoder states

		if self.self_attn:
			
			# self attention on encoder and decoder states
			enc_self_attn_op, _, _ = self.enc_self_attention(enc_outputs, enc_outputs, enc_outputs)
			dec_outputs = torch.cat((dec_prev_outputs[:,1:,:], lstm_output), 1)
			dec_self_attn_op, _, _ = self.dec_self_attention(dec_outputs, dec_outputs, dec_outputs)

			# intra attention between encoder and decoder states (use intra-temporal attention only if enc_prev_attn_scores are not none)
			# Note: intra-temporal attention not supported for multiplicative attention as training becomes unstable
			enc_context, enc_attn_weights, enc_attn_scores = self.enc_attention(final_hidden[-1], enc_self_attn_op, enc_prev_attn_scores)
			dec_context, dec_attn_weights, dec_attn_scores = self.dec_attention(final_hidden[-1], dec_self_attn_op[:,:-1,:])
			context = torch.cat((enc_context, dec_context), 2)
		else:
			# without self attention if dec_attn is also true then the hidden state at time step t will also attend to prev decoder states
			context, enc_attn_weights, enc_attn_scores = self.enc_attention(final_hidden[-1], enc_outputs, enc_prev_attn_scores)
			if self.dec_attn:
				dec_outputs = torch.cat((dec_prev_outputs[:,1:,:], lstm_output),1)
				dec_context, dec_attn_weights, dec_attn_scores = self.dec_attention(final_hidden[-1], dec_outputs)
				context = torch.cat((context, dec_context), 2)


		logits = self.out(torch.cat((lstm_output, context), 2)).squeeze(1)


		output = F.log_softmax(logits, dim=1) # [Bxoutput_size]

		return output, lstm_output, (final_hidden, final_cell) , enc_attn_weights, enc_attn_scores

	def init_hidden_and_cell(self, enc_final_hidden, batch_size):

		hidden = torch.tanh(self.enc2dec_hidden(enc_final_hidden)) 
		cell = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
		return (hidden[:self.num_layers], cell)


class Seq2Seq(nn.Module):

	def __init__(self, encoder, decoder):

		super(Seq2Seq, self).__init__()
		self.encoder = encoder
		self.decoder = decoder


	def forward(self, source, target, lengths):

		# source and target are [BxT] tensors
		batch_size = source.size(0)
		max_T = target.size(1) # max seq len
		enc_output, hidden = self.encoder(source, lengths)
		states = self.decoder.init_hidden_and_cell(hidden, batch_size)
		outputs = torch.zeros(batch_size, max_T, self.decoder.output_size, device=device)
		
		dec_prev_outputs = None
		enc_prev_attn_scores = None
		
		if self.decoder.dec_attn:
			dec_prev_outputs = torch.zeros(batch_size, 1, self.decoder.hidden_size, device=device)
		
		if self.decoder.intra_temp_attn:
			enc_prev_attn_scores = torch.zeros(batch_size, enc_output.size(1), 1, device=device) #[BxTxx1]
		
		dec_input = target[:,0].view(-1,1)

		# unroll the decoder one step at a time
		for i in range(max_T):

			dec_output, output_hidden, states, attn_weights, enc_attn_scores = self.decoder(dec_input, enc_output, states[0], states[1], dec_prev_outputs, enc_prev_attn_scores)
			
			if self.decoder.dec_attn:
				dec_prev_outputs = torch.cat((dec_prev_outputs,output_hidden),1)
			
			if self.decoder.intra_temp_attn:
				enc_prev_attn_scores = torch.cat((enc_prev_attn_scores, enc_attn_scores), 2)

			outputs[:,i,:] = dec_output
			# dec_top_pred = torch.argmax(dec_output, 1) # get the indices corresponding to max prob output
			#next_input = target[:,i].view(-1,1) if random.random()<teacher_forcing_ratio else dec_top_pred.view(-1,1)
			next_input = target[:,i].view(-1,1)
			dec_input = next_input

		return outputs

	def greedy_decode(self, source, lengths, go_idx, max_len=50):

		outputs = []
		attns = []
		batch_size = source.size(0)

		with torch.no_grad(): # set all required_grad's to false
			enc_output, hidden = self.encoder(source, lengths)
			states = self.decoder.init_hidden_and_cell(hidden, batch_size)
		
			dec_prev_outputs = None
			enc_prev_attn_scores = None
		
			if self.decoder.dec_attn:
				dec_prev_outputs = torch.zeros(batch_size, 1, self.decoder.hidden_size, device=device)
			
			if self.decoder.intra_temp_attn:
				enc_prev_attn_scores = torch.zeros(batch_size, enc_output.size(1), 1, device=device)

			dec_input = torch.tensor([[go_idx]*batch_size], device=device)
			dec_input = torch.transpose(dec_input, 0, 1)
			# print(dec_input.shape)

			for i in range(max_len):

				dec_output, output_hidden, states, attn_weights, enc_attn_scores = self.decoder(dec_input, enc_output, states[0], states[1], dec_prev_outputs, enc_prev_attn_scores)

				if self.decoder.dec_attn:				
					dec_prev_outputs = torch.cat((dec_prev_outputs,output_hidden),1)
				
				if self.decoder.intra_temp_attn:				
					enc_prev_attn_scores = torch.cat((enc_prev_attn_scores, enc_attn_scores), 2)

				dec_top_pred = torch.argmax(dec_output,1)
				dec_input = dec_top_pred.view(-1,1)
				attn_weights = attn_weights.squeeze(2)
				outputs.append(dec_top_pred.data.cpu().numpy())
				attns.append(attn_weights.data.cpu().numpy())

		outputs = np.transpose(np.array(outputs), (1,0)) # [BxTy]
		attns = np.transpose(np.array(attns), (1,0,2)) # [BxTyxTx]
		
		return outputs, attns 










