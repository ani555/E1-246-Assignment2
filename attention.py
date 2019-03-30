
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdditiveAttention(nn.Module):

	def __init__(self, query_size, val_size, proj_size=128):

		super(AdditiveAttention, self).__init__()
		self.query_size = query_size
		self.val_size = val_size
		self.proj_size = proj_size
		self.query_layer = nn.Linear(query_size, proj_size, bias=False)
		self.val_layer = nn.Linear(val_size, proj_size, bias=False)
		self.energy_layer = nn.Linear(proj_size, 1, bias=False)


	def forward(self, query, values, enc_prev_attn_scores=None):

		# Bahdanau attention v.T*tanh(W1*h + W2*q)
		# values are encoder final hidden states
		# query is decoder last hidden state

		val_proj = self.val_layer(values) # [BxTxP]
		query_proj = self.query_layer(query) # [BxP]
		query_proj = query_proj.unsqueeze(1) # [Bx1xP]
		attn_scores = self.energy_layer(torch.tanh(val_proj+query_proj)) # [BxTx1]
		if enc_prev_attn_scores is not None:
			# intra-temporal attention on input sequence (refer Paulus et. al (2017) paper)
			t = enc_prev_attn_scores.size(-1)

			if t==1:
				attn_scores_dash = torch.exp(attn_scores)
			else:
				attn_scores_dash = torch.exp(attn_scores)/torch.sum(torch.exp(enc_prev_attn_scores[:,:,1:]), dim=-1, keepdim=True)		
		
		attn_scores = torch.transpose(attn_scores, 1, 2) # [Bx1xT]
		
		if enc_prev_attn_scores is not None:
			attn_scores_dash = torch.transpose(attn_scores_dash, 1, 2) # [Bx1xT]
			attn_outputs = attn_scores_dash/torch.sum(attn_scores_dash, dim=-1, keepdim=True)
		else:
			attn_outputs = F.softmax(attn_scores, dim=-1)
		
		context = torch.bmm(attn_outputs, values)
		attn_outputs = torch.transpose(attn_outputs, 1, 2) # [BxTx1]
		attn_scores = torch.transpose(attn_scores, 1, 2)
		return context, attn_outputs, attn_scores


class MultiplicativeAttention(nn.Module):

	def __init__(self, query_size, val_size):

		super(MultiplicativeAttention, self).__init__()
		self.query_size = query_size
		self.val_size = val_size
		self.val_layer = nn.Linear(val_size, query_size, bias=False)

	def forward(self, query, values, enc_prev_attn_scores=None):
			
		# Luong attention q.T*W*h
		val_proj = self.val_layer(values) # [BxTxH]
		query = query.unsqueeze(2)  # [BxHx1]
		attn_scores = torch.bmm(val_proj, query) # [BxTx1]
		
		attn_scores = torch.transpose(attn_scores, 1, 2) # [Bx1xT]
		
		attn_outputs = F.softmax(attn_scores, dim=-1)
		context = torch.bmm(attn_outputs, values)
		attn_outputs = torch.transpose(attn_outputs, 1, 2)
		attn_scores = torch.transpose(attn_scores, 1, 2)
		return context, attn_outputs, attn_scores

class ScaledDotProductAttention(nn.Module):

	"""
	Scaled dot product attention is same as multiplicative attention with an additional scaling factor dk (key_size). For large values 
	of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients
	- Attention is all you need (Vaswani et. al)
	"""

	def __init__(self, query_size, val_size):

		super(ScaledDotProductAttention, self).__init__()
		self.query_size = query_size
		self.val_size = val_size # key_size = val_size
		self.val_layer = nn.Linear(val_size, query_size, bias=False) # here keys are same as values

	def forward(self, query, values, enc_prev_attn_scores=None):

		val_proj = self.val_layer(values) # [BxTxH]
		query = query.unsqueeze(2)  # [BxHx1]
		attn_scores = torch.bmm(val_proj, query)/math.sqrt(val_proj.size(-1)) # [BxTx1]
		if enc_prev_attn_scores is not None:
			t = enc_prev_attn_scores.size(-1)
			if t==1:
				attn_scores_dash = torch.exp(attn_scores)
			else:
				attn_scores_dash = torch.exp(attn_scores)/torch.sum(torch.exp(enc_prev_attn_scores[:,:,1:]), dim=-1, keepdim=True)		
		
		attn_scores = torch.transpose(attn_scores, 1, 2) # [Bx1xT]
		
		if enc_prev_attn_scores is not None:
			attn_scores_dash = torch.transpose(attn_scores_dash, 1, 2) # [Bx1xT]
			attn_outputs = attn_scores_dash/torch.sum(attn_scores_dash, dim=-1, keepdim=True)
		else:
			attn_outputs = F.softmax(attn_scores, dim=-1)
		
		context = torch.bmm(attn_outputs, values)
		attn_outputs = torch.transpose(attn_outputs, 1, 2)
		attn_scores = torch.transpose(attn_scores, 1, 2)
		return context, attn_outputs, attn_scores

class KeyValueAttention(nn.Module):

	# Reference paper - Frustratingly short attention spans in Neural Language Modelling- Daniluk et. al
	def __init__(self, query_size, key_size, val_size, proj_size):

		super(KeyValueAttention, self).__init__()
		self.query_size = query_size
		self.key_size = key_size
		self.val_size = val_size
		self.proj_size = proj_size
		self.query_key_layer = nn.Linear(query_size, proj_size, bias=False)
		self.query_val_layer = nn.Linear(query_size, proj_size, bias=False)
		self.key_layer = nn.Linear(key_size, proj_size, bias=False)
		self.val_layer = nn.Linear(val_size, proj_size, bias=False)
		self.Wy_layer = nn.Linear(proj_size, proj_size, bias=False)
		self.Wh_layer = nn.Linear(proj_size, proj_size, bias=False)
		self.energy_layer = nn.Linear(proj_size, 1, bias=False)

	def forward(self, query, outputs, enc_prev_attn_scores=None):

		query = query.unsqueeze(1)
		query_key_proj = self.query_key_layer(query) # [Bx1Xk] 
		query_val_proj = self.query_val_layer(query) # [Bx1xk]
		output_key_proj = self.key_layer(outputs) # [BxTxK]
		output_val_proj = self.val_layer(outputs) # [BxTxK]
		ones = torch.ones(output_key_proj.size(0), output_key_proj.size(1), 1, device=device) # [BxTX1]		
		M_t = torch.tanh(self.Wy_layer(output_key_proj) + torch.bmm(ones, self.Wh_layer(query_key_proj))) # [BxTxK]
		attn_scores = self.energy_layer(M_t)

		if enc_prev_attn_scores is not None:
			t = enc_prev_attn_scores.size(-1)
			if t==1:
				attn_scores_dash = torch.exp(attn_scores)
			else:
				attn_scores_dash = torch.exp(attn_scores)/torch.sum(torch.exp(enc_prev_attn_scores[:,:,1:]), dim=-1, keepdim=True)		
		
		attn_scores = torch.transpose(attn_scores, 1, 2) # [Bx1xT]
		
		if enc_prev_attn_scores is not None:
			attn_scores_dash = torch.transpose(attn_scores_dash, 1, 2) # [Bx1xT]
			attn_outputs = attn_scores_dash/torch.sum(attn_scores_dash, dim=-1, keepdim=True)
		else:
			attn_outputs = F.softmax(attn_scores, dim=-1)
		
		context = torch.bmm(attn_outputs, output_val_proj)
		attn_outputs = torch.transpose(attn_outputs, 1, 2)
		attn_scores = torch.transpose(attn_scores, 1, 2)

		return context, attn_outputs, attn_scores

class SelfAttention(nn.Module):

	def __init__(self, query_size, key_size, val_size, proj_size):

		super(SelfAttention, self).__init__()
		self.query_size = query_size
		self.key_size = key_size
		self.val_size = val_size
		self.proj_size = proj_size
		self.query_layer = nn.Linear(query_size, proj_size, bias=False)
		self.key_layer = nn.Linear(key_size, proj_size, bias=False)
		#self.val_layer = nn.Linear(val_size, proj_size, bias=False)

	def forward(self, querys, keys, values):

		proj_querys = self.query_layer(querys) #[BxTxP]
		proj_keys = self.key_layer(keys) #[BxTxP]
		#proj_vals = self.val_layer(values) #[BxTxP]
		proj_keys = torch.transpose(proj_keys, 1, 2) #[BxPxT]
		attn_scores = torch.bmm(proj_querys, proj_keys) / math.sqrt(self.proj_size) # [BxTxT]
		attn_outputs = F.softmax(attn_scores, dim=-1)
		context = torch.bmm(attn_outputs, values) #[BxTxH]

		return context, attn_outputs, attn_scores 
