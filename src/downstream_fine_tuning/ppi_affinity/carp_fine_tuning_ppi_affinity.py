import numpy as np
import pandas as pd
import sys
import copy
import pickle
import time 
import os

import json 

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats 
from config import huggingface_cache_dir, results_path, carp_model_path
import re

from sequence_models.pretrained import load_model_and_alphabet



class LinearHead(nn.Module):
	def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.0, pool_method = "cls_pooled",  dual_handling = "averaged"):
		super().__init__()
		self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
		self.out_proj = nn.Linear(hidden_size, num_labels)
		self.pool_method = pool_method
		self.dual_handling = dual_handling

	def forward(self, features_1, attention_mask_1, features_2, attention_mask_2):

		# features are [batch, length with padding , hidden]
		
		if features_1.dim() == 3:
			if self.pool_method == "cls_pooled":
				features_1 = features_1[:, 0, :]   # taking CLS token gives [batch, hidden]

			elif self.pool_method == "mean_pooled":
				if attention_mask_1 is None:
					raise ValueError("attention_mask required for mean pooling")

				# expand mask to match hidden size
				mask_1 = attention_mask_1.unsqueeze(-1).to(features_1.dtype) # [batch, seq_len, 1]

				# zero out padding tokens
				masked_features_1 = features_1 * mask_1

				# sum over sequence
				summed_1 = masked_features_1.sum(dim=1)  # [batch, hidden]

				# count valid tokens
				lengths_1 = mask_1.sum(dim=1)  # [batch, 1], these will match the actual seqeunce lenghts 
				# avoid division by zero
				valid_counts_1 = mask_1.sum(dim=1).clamp(min=1e-9)  # [batch, 1]
				
				features_1 = summed_1 / valid_counts_1  # [batch, hidden]


		if features_2.dim() == 3:
			if self.pool_method == "cls_pooled":
				features_2 = features_2[:, 0, :]   # taking CLS token gives [batch, hidden]

			elif self.pool_method == "mean_pooled":
				if attention_mask_2 is None:
					raise ValueError("attention_mask required for mean pooling")

				# expand mask to match hidden size
				mask_2 = attention_mask_2.unsqueeze(-1).to(features_2.dtype) # [batch, seq_len, 1]

				# zero out padding tokens
				masked_features_2 = features_2 * mask_2

				# sum over sequence
				summed_2 = masked_features_2.sum(dim=1)  # [batch, hidden]

				# count valid tokens
				lengths_2 = mask_2.sum(dim=1)  # [batch, 1], these will match the actual seqeunce lenghts 

				# avoid division by zero
				valid_counts_2 = mask_2.sum(dim=1).clamp(min=1e-9)  # [batch, 1]
				
				features_2 = summed_2 / valid_counts_2  # [batch, hidden]
		
		
		if self.dual_handling == "averaged" :
			features = (features_1 + features_2) / 2 # [batch, hidden_dim] 
		else:
			features = torch.cat([features_1, features_2], dim=1)


		x = self.dropout(features) 
		x = self.out_proj(x).squeeze(-1) # [batch, hidden] -> [batch, num classes] 

		return x


class RegModel(nn.Module):
	def __init__(
		self, 
		plm,
		hidden_size, 
		num_labels, 
		dropout, 
		pool_method = "cls_pooled",
		dual_handling = "averaged"
	):
		super().__init__()
		self.plm = plm
		self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
		self.num_labels = num_labels
		self.pool_method = pool_method
		self.linear_head = LinearHead(hidden_size, num_labels, dropout, pool_method, dual_handling = dual_handling)


	def forward(self, x):
		input_ids_1 = x["input_ids_1"]
		attention_mask_1 = x["attention_mask_1"]

		input_ids_2 = x["input_ids_2"]
		attention_mask_2 = x["attention_mask_2"]

		rep_1 = self.plm(input_ids_1, repr_layers=[layer_num], logits=True)
		rep_1 = rep_1["representations"][layer_num]

		rep_2 = self.plm(input_ids_2, repr_layers=[layer_num], logits=True)
		rep_2 = rep_2["representations"][layer_num]

		# print("point 1", x.size())
		# models with 

		outputs = self.linear_head(rep_1, attention_mask_1, rep_2, attention_mask_2)

		# print("point -1", x.size())
		return outputs




class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return super().default(obj)

class SeqRegDataset(Dataset):
	def __init__(self, df):
		self.seqs_1 = df["sequence_1"].tolist()
		self.seqs_2 = df["sequence_2"].tolist()

		self.labels = df["label"].tolist()
		self.identifiers = df["id"].tolist()

		self.lengths_1 = list(map(lambda x : len(x), self.seqs_1))
		self.lengths_2 = list(map(lambda x : len(x), self.seqs_2))

	def __len__(self):
		return len(self.seqs_1)

	def __getitem__(self, idx):
		return {
			"sequences_1": self.seqs_1[idx], 
			"sequences_2": self.seqs_2[idx], 
			"labels": self.labels[idx], 
			"identifiers" : self.identifiers[idx], 
			"lengths_1" : self.lengths_1[idx],
			"lengths_2" : self.lengths_2[idx]
		}


def prep_prott5_sequence(seq):
	seq = seq.upper()
	seq = re.sub(r"[UZOB]", "X", seq)
	seq = " ".join(list(seq))
	return seq

def collate_fn(examples):
	sequences_1 = [ex["sequences_1"] for ex in examples]
	sequences_2 = [ex["sequences_2"] for ex in examples]

	labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.float32)
	identifiers = [ex["identifiers"] for ex in examples]  # keep as python list
	lengths_1 = [ex["lengths_1"] for ex in examples]
	lengths_2 = [ex["lengths_2"] for ex in examples]

	seq_list_1 = [[s] for s in sequences_1]
	seq_list_2 = [[s] for s in sequences_2]

	# print(seq_list)
	# seqs = [["MDREQ"], ["MGTRRLLP"]]
	input_ids_1 = tokenizer(seq_list_1,)[0]
	input_ids_2 = tokenizer(seq_list_2,)[0]

	# attention mask is everything thats 27 
	attention_mask_1 = (input_ids_1 != 27).long()
	attention_mask_2 = (input_ids_2 != 27).long()

	batch = {}

	batch["input_ids_1"] = input_ids_1
	batch["attention_mask_1"] = attention_mask_1

	batch["input_ids_2"] = input_ids_2
	batch["attention_mask_2"] = attention_mask_2
	# batch = data_collator(batch)  # pads input_ids/attention_mask to max in batch
	batch["labels"] = labels
	batch["identifiers"] = identifiers
	batch["lengths_1"] = lengths_1
	batch["lengths_2"] = lengths_1
	
	return batch


if __name__ == "__main__":
	global max_seq_len
	global model_key
	global layer_num
	global tokenizer

	max_seq_len = 1022

	model_key = sys.argv[1]
	pool_method = sys.argv[2]
	dual_handling = sys.argv[3]
	random_state = int(sys.argv[4])

	prefix = "mar_31"
	out_fn = f"{results_path}ppi_affinity/{prefix}_{model_key}_{pool_method}_random_state_{random_state}_{dual_handling}.json"
	ckpt_fn = f"{results_path}ppi_affinity/ckpt_{prefix}_{model_key}_{pool_method}_random_state_{random_state}_{dual_handling}.json"
	print(out_fn)

	model_key = sys.argv[1] # carp_600k, carp_38M, carp_76M, carp_640M

	
	if pool_method == "cls_pooled" and model_key in ["prott5_xl", "prott5_xl_bfd"]:
		print("no CLS token in prott5 model")
		quit()

	torch.manual_seed(random_state)  


	data = pd.read_csv("../../review_fine_tuning/data/ppi_affinity_data.csv")
	data["length_1"] = data["sequence_1"].apply(lambda x : len(x))
	data["length_2"] = data["sequence_2"].apply(lambda x : len(x))
	data = data.rename(columns = {"score":  "label"})

	data = data.loc[
		(data["length_1"] <= 1022) & 
		(data["length_2"] <= 1022) 
	]

	train_data = data.loc[data["partition"] == "train"] 
	validation_data = data.loc[data["partition"] == "val"] 
	test_data = data.loc[data["partition"] == "test"] 

	print("train :" , len(train_data), "val", len(validation_data), "test", len(test_data))

	
	
	################################
	# CARP stuff


	plm, tokenizer = load_model_and_alphabet(f'{carp_model_path}{model_key}.pt')
	model_data = torch.load(f'{carp_model_path}{model_key}.pt', map_location="cpu")

	device = "cuda" if torch.cuda.is_available() else "cpu"
	plm.to(device)


	# alternatively : from sequence_models.constants import PROTEIN_ALPHABET, PAD, MASK, GAP , Start, etc. 
	CAN_AAS = 'ACDEFGHIKLMNPQRSTVWY'
	AMB_AAS = 'BZX'
	OTHER_AAS = 'JOU'
	ALL_AAS = CAN_AAS + AMB_AAS + OTHER_AAS
	STOP = '*'
	GAP = '-'
	PAD = GAP
	MSA_PAD = '!'
	MASK = '#'  # Useful for masked language model training
	START = '@'
	SEP = '/'
	SPECIALS = STOP + GAP + MASK + START
	PROTEIN_ALPHABET = ALL_AAS + SPECIALS
	#####

	index_to_char = {}
	char_to_index = {}

	for i, char in enumerate([*PROTEIN_ALPHABET]):
		index_to_char[i] = char
		char_to_index[char] = i

	mask_idx = PROTEIN_ALPHABET.index(MASK)

	d_embedding = model_data['d_embed']
	d_model = model_data['d_model']
	
	layer_num = model_data['n_layers']

	kernel_size = model_data['kernel_size']
	activation = model_data['activation']
	slim = model_data['slim']
	r = model_data['r']

	bgc = 'bigcarp' in model_data['model']
	if not bgc:
		n_tokens = len(PROTEIN_ALPHABET)
		mask_idx = PROTEIN_ALPHABET.index(MASK)
		pad_idx = PROTEIN_ALPHABET.index(PAD)
		n_frozen = None


	#############################

	hidden_size = d_model

	if dual_handling == "concatenated":
		hidden_size = 2*hidden_size


	eff_batch_size = 64

	train_bs = 16
	eval_bs = 16

	if model_key == "carp_640M":
		train_bs = 8
		eval_bs = 8
	

	gradient_accumulation_steps = eff_batch_size // train_bs


	train_ds = SeqRegDataset(train_data)
	val_ds   = SeqRegDataset(validation_data)
	test_ds  = SeqRegDataset(test_data)	


	train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=False,  collate_fn=collate_fn)
	val_loader   = DataLoader(val_ds,   batch_size=eval_bs,  shuffle=False,  collate_fn=collate_fn)
	test_loader  = DataLoader(test_ds,  batch_size=eval_bs,  shuffle=False,  collate_fn=collate_fn)

	

	model = RegModel(plm, hidden_size, 1, 0.0, pool_method = pool_method, dual_handling = dual_handling)

	
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model.to(device)

	
	loss_fn = nn.MSELoss()


	total_train_steps = 0 # number of optimizer calls 
	total_steps = 0 # total passes

	n_epochs = 100


	total_train_steps = 0

	MAX_TRAIN_STEPS = 10000
	EVAL_TRAIN_STEPS = 100


	results = {}


	LAYERS_TO_TRAIN = 4

	for layer in model.plm.model.embedder.layers[:-1*(LAYERS_TO_TRAIN)]:
		for param in layer.parameters():
			param.requires_grad = False


	lr = 1e-5
	weight_decay = 0.01
	
	optimizer = torch.optim.AdamW(
		[p for p in model.parameters() if p.requires_grad],
		lr=lr, 
		weight_decay=weight_decay
	)


	for epoch in range(n_epochs):
		t1 = time.time()

		model.train()

		running_loss = 0.0
		for batch in train_loader:

			total_steps += 1

			lengths_1 = list(batch.pop("lengths_1")) # dont pass this to the model
			lengths_2 = list(batch.pop("lengths_2")) # dont pass this to the model
			identifiers = list(batch.pop("identifiers")) # dont pass this to the model
			labels = batch.pop("labels").to(device)

			# batch contains dict with input_ids and attention_mask

			# move only tensor entries to device
			batch = {k: v.to(device) for k, v in batch.items()}

			optimizer.zero_grad()
			output = model(batch)

			loss = loss_fn(output, labels)

			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			if total_steps % gradient_accumulation_steps == 0 and total_steps > 0:
				optimizer.step()
				optimizer.zero_grad()

				total_train_steps += 1

				print("train steps ", total_train_steps)

			if ((total_train_steps % EVAL_TRAIN_STEPS == 0) and (total_train_steps > 0)) or total_train_steps == MAX_TRAIN_STEPS:
				
				total_train_steps += 1
				t2 = time.time()

				print("elapsed", total_train_steps,  round(t2 -t1, 4))
				
				model.eval()
				
				with torch.no_grad():

					all_identities = []
					all_true_labels = []
					all_predicted_labels = []
					running_loss = 0 
					for batch in val_loader:
						lengths_1 = list(batch.pop("lengths_1")) # dont pass this to the model
						lengths_2 = list(batch.pop("lengths_2")) # dont pass this to the model
						identifiers = list(batch.pop("identifiers")) # dont pass this to the model
						labels = batch.pop("labels").to(device)

						# move only tensor entries to device
						batch = {k: v.to(device) for k, v in batch.items()}

						optimizer.zero_grad()
						output = model(batch)

						loss = loss_fn(output, labels)
						running_loss += loss.item()



						all_identities.extend(list(identifiers))  # if idents are strings/ints/etc.
						all_true_labels.extend(labels.detach().cpu().tolist())
						all_predicted_labels.extend(output.detach().cpu().tolist())



				val_json = {
					"identifier" : all_identities,
					"true_label" : all_true_labels,
					"predicted_label" : all_predicted_labels
				}

				validataion_results = {"loss" : running_loss, "results" : val_json}


				with torch.no_grad():
				
					all_identities = []
					all_true_labels = []
					all_predicted_labels = []

					running_loss = 0 
					for batch in test_loader:
						lengths_1 = list(batch.pop("lengths_1")) # dont pass this to the model
						lengths_2 = list(batch.pop("lengths_2")) # dont pass this to the model

						identifiers = list(batch.pop("identifiers")) # dont pass this to the model
						labels = batch.pop("labels").to(device)

						# move only tensor entries to device
						batch = {k: v.to(device) for k, v in batch.items()}

						optimizer.zero_grad()
						output = model(batch)

						loss = loss_fn(output, labels)
						running_loss += loss.item()


						all_identities.extend(list(identifiers))  # if idents are strings/ints/etc.
						all_true_labels.extend(labels.detach().cpu().tolist())
						all_predicted_labels.extend(output.detach().cpu().tolist())
						



					test_json = {
						"identifier" : all_identities,
						"true_label" : all_true_labels,
						"predicted_label" : all_predicted_labels
					}

					test_results = {"loss" : running_loss, "results" : test_json}



				results[total_train_steps] = {"validation" : validataion_results, "test": test_results}

				with open(ckpt_fn, "w") as f1:
					json.dump(results, f1, cls=NumpyEncoder)
				f1.close()

				t1 = time.time()

				model.train()

			if total_train_steps == MAX_TRAIN_STEPS:
				break

		if total_train_steps == MAX_TRAIN_STEPS:
			break

	with open(out_fn, "w") as f1:
		json.dump(results, f1, cls=NumpyEncoder)
	f1.close()

	print("done", out_fn)



		

