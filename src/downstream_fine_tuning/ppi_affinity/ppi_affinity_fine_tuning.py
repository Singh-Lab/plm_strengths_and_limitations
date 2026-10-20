import numpy as np
import pandas as pd
import sys
import copy
import pickle
import time 
import os
from transformers import (
	EsmTokenizer,
	EsmModel,
	BertTokenizer,
	BertModel,
	EsmForSequenceClassification,
	DataCollatorWithPadding,
	T5Tokenizer,
	T5Model,
	T5EncoderModel,

)
import json 

from sklearn.metrics import f1_score, accuracy_score

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats 
from config import huggingface_cache_dir, results_path
import re





class LinearHead(nn.Module):
	def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.0, pool_method = "cls_pooled", dual_handling = "average"):
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
				mask = attention_mask_1.unsqueeze(-1).clone()  # [batch, seq_len, 1]
				mask = mask.to(features_1.dtype) # need to do this for multiplication 

				# --- remove CLS --- 
				if "esm" in model_key or "protbert" in model_key: # no CLS in prott5 models
					mask[:, 0] = 0

				# --- remove EOS ---
				# find last non-padding token per sequence
				lengths = attention_mask_1.sum(dim=1)  # [batch], these will be one longer than the seqeunce, just to get the index of the eos token

				# print(lengths.size())
				
				# subtract 1 to get EOS index
				eos_indices = lengths - 1  # [batch],

				# zero out EOS positions
				batch_indices = torch.arange(mask.size(0), device=mask.device)
				mask[batch_indices, eos_indices, :] = 0 # changed this line from mask[batch_indices, eos_indices]

				# zero out padding/special tokens
				masked_features = features_1 * mask

				# sum over sequence
				summed = masked_features.sum(dim=1)  # [batch, hidden]

				# count valid tokens
				lengths = mask.sum(dim=1)  # [batch, 1], these will match the actual seqeunce lenghts 

				# avoid division by zero
				valid_counts = mask.sum(dim=1).clamp(min=1e-9)  # [batch, 1]
				features_1 = summed / valid_counts  # [batch, hidden]

		if features_2.dim() == 3:
			if self.pool_method == "cls_pooled":
				features_2 = features_2[:, 0, :]   # taking CLS token gives [batch, hidden]

			elif self.pool_method == "mean_pooled":
				if attention_mask_1 is None:
					raise ValueError("attention_mask required for mean pooling")

				# expand mask to match hidden size
				mask = attention_mask_2.unsqueeze(-1).clone()  # [batch, seq_len, 1]
				mask = mask.to(features_2.dtype) # need to do this for multiplication 

				# --- remove CLS --- 
				if "esm" in model_key or "protbert" in model_key: # no CLS in prott5 models
					mask[:, 0] = 0

				# --- remove EOS ---
				# find last non-padding token per sequence
				lengths = attention_mask_2.sum(dim=1)  # [batch], these will be one longer than the seqeunce, just to get the index of the eos token

				# print(lengths.size())
				
				# subtract 1 to get EOS index
				eos_indices = lengths - 1  # [batch],

				# zero out EOS positions
				batch_indices = torch.arange(mask.size(0), device=mask.device)
				mask[batch_indices, eos_indices, :] = 0 # changed this line from mask[batch_indices, eos_indices]

				# zero out padding/special tokens
				masked_features = features_2 * mask

				# sum over sequence
				summed = masked_features.sum(dim=1)  # [batch, hidden]

				# count valid tokens
				lengths = mask.sum(dim=1)  # [batch, 1], these will match the actual seqeunce lenghts 

				# avoid division by zero
				valid_counts = mask.sum(dim=1).clamp(min=1e-9)  # [batch, 1]
				features_2 = summed / valid_counts  # [batch, hidden]
			

		if self.dual_handling == "averaged" :
			features = (features_1 + features_2) / 2 # [batch, hidden_dim] 
		else:
			features = torch.cat([features_1, features_2], dim=1)

		x = self.dropout(features) 
		x = self.out_proj(x).squeeze(-1) # [batch, hidden] -> [batch] 

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
		# can get input ids with : input_ids = x["input_ids"]
		attention_mask_1 = x["attention_mask_1"]
		attention_mask_2 = x["attention_mask_2"]
		
		input_ids_1 = x["input_ids_1"]
		input_ids_2 = x["input_ids_2"]

		rep_1 = self.plm(input_ids = input_ids_1, attention_mask = attention_mask_1).last_hidden_state
		rep_2 = self.plm(input_ids = input_ids_2, attention_mask = attention_mask_2).last_hidden_state
	
		x = self.linear_head(rep_1, attention_mask_1, rep_2, attention_mask_2)
		return x




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



def collate_fn(examples):
	sequences_1 = [ex["sequences_1"] for ex in examples]
	sequences_2 = [ex["sequences_2"] for ex in examples]

	labels_raw = [ex["labels"] for ex in examples]

	labels = torch.tensor([ex["labels"] for ex in examples])
	identifiers = [ex["identifiers"] for ex in examples]  # keep as python list
	lengths_1 = [ex["lengths_1"] for ex in examples]
	lengths_2 = [ex["lengths_2"] for ex in examples]

	if "prott5" in model_key or "protbert" in model_key:
		sequences_1 = [prep_prott5_sequence(seq) for seq in sequences_1]
		sequences_2 = [prep_prott5_sequence(seq) for seq in sequences_2]

	batch_1 = tokenizer(
		sequences_1,
		truncation=True,
		max_length=max_seq_len,
		padding=False,   # collator will pad
		return_tensors=None,
	)

	batch_2 = tokenizer(
		sequences_2,
		truncation=True,
		max_length=max_seq_len,
		padding=False,   # collator will pad
		return_tensors=None,
	)

	batch_1 = data_collator(batch_1)  # pads input_ids/attention_mask to max in batch
	batch_2 = data_collator(batch_2)  # pads input_ids/attention_mask to max in batch

	batch = {}

	batch["input_ids_1"] = batch_1["input_ids"]
	batch["attention_mask_1"] = batch_1["attention_mask"]
	batch["input_ids_2"] = batch_2["input_ids"]
	batch["attention_mask_2"] = batch_2["attention_mask"]
	batch["labels"] = labels
	batch["identifiers"] = identifiers
	batch["lengths_1"] = lengths_1
	batch["lengths_2"] = lengths_2

	return batch

def prep_prott5_sequence(seq):
	seq = seq.upper()
	seq = re.sub(r"[UZOB]", "X", seq)
	seq = " ".join(list(seq))
	return seq


if __name__ == "__main__":
	global max_seq_len
	global model_key

	max_seq_len = 1022

	model_key = sys.argv[1]
	pool_method = sys.argv[2]
	dual_handling = sys.argv[3]
	random_state = int(sys.argv[4])

	prefix = "mar31"

	out_fn = f"{results_path}ppi_affinity/{prefix}_{model_key}_{pool_method}_{dual_handling}_random_state_{random_state}.json"
	ckpt_fn = f"{results_path}ppi_affinity/ckpt_{prefix}_{model_key}_{pool_method}_{dual_handling}_random_state_{random_state}.json"

	print(out_fn)


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


	print("max length", np.max(data["length_1"].values),np.max(data["length_2"].values) )

	train_data = data.loc[data["partition"] == "train"] 
	validation_data = data.loc[data["partition"] == "val"] 
	test_data = data.loc[data["partition"] == "test"] 

	print("train :" , len(train_data), "val", len(validation_data), "test", len(test_data))

	key_to_model_string = {
		"esm_1b" : "facebook/esm1b_t33_650M_UR50S",
		"esm_2_3B" :  "facebook/esm2_t36_3B_UR50D",
		"esm_1v" : "facebook/esm1v_t33_650M_UR90S_1",
		"esm_2_15B" : "facebook/esm2_t48_15B_UR50D",
		"esm_2_650M" : "facebook/esm2_t33_650M_UR50D",
		"esm_2_150M" : "facebook/esm2_t30_150M_UR50D",
		"esm_2_35M" :  "facebook/esm2_t12_35M_UR50D",
		"esm_2_8M" :   "facebook/esm2_t6_8M_UR50D",
		"protbert" : "Rostlab/prot_bert",
		"protbert_bfd" : "Rostlab/prot_bert_bfd",
		"prott5_xl" : "Rostlab/prot_t5_xl_uniref50", # recommended to use the encoder layer by the Rost Lab
		"prott5_xl_bfd" : "Rostlab/prot_t5_xl_bfd", # recommended to use the encoder layer by the Rost Lab
	}

	model_name = key_to_model_string[model_key]

	if "esm" in model_key:
		tokenizer = EsmTokenizer.from_pretrained(
			model_name,
			do_lower_case=False,
			cache_dir = huggingface_cache_dir,
			return_tensors="pt", 
			padding=True, 
			truncation=True
		)

		plm = EsmModel.from_pretrained(
			model_name,
			num_labels=2, # Solubility has 2 classes
			ignore_mismatched_sizes=True,
			cache_dir = huggingface_cache_dir
		)

	elif model_key in ["protbert", "protbert_bfd"]:
		tokenizer = BertTokenizer.from_pretrained(
			model_name,
			do_lower_case=False,
			cache_dir=huggingface_cache_dir,
			return_tensors="pt", 
			padding=True, 
			truncation=True
		)

		plm = BertModel.from_pretrained(model_name, cache_dir=huggingface_cache_dir)



	elif model_key in ["prott5_xl", "prott5_xl_bfd"]:

		tokenizer = T5Tokenizer.from_pretrained(
			model_name,
			do_lower_case=False,
			cache_dir=huggingface_cache_dir,
			return_tensors="pt", 
			padding=True, 
			truncation=True
		)

		plm = T5EncoderModel.from_pretrained(
			model_name,
			cache_dir=huggingface_cache_dir
		)

	eff_batch_size = 64
	train_bs = 8
	eval_bs = 8
	LAYERS_TO_TRAIN = 4

	if "prott5" in model_key or model_key == "esm_2_3B":
		train_bs = 4
		eval_bs = 4

	gradient_accumulation_steps = eff_batch_size // train_bs

	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

	train_ds = SeqRegDataset(train_data)
	val_ds   = SeqRegDataset(validation_data)
	test_ds  = SeqRegDataset(test_data)	


	train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=False,  collate_fn = collate_fn)
	val_loader   = DataLoader(val_ds,   batch_size=eval_bs,  shuffle=False, collate_fn = collate_fn)
	test_loader  = DataLoader(test_ds,  batch_size=eval_bs,  shuffle=False, collate_fn = collate_fn)

	hidden_size = plm.config.hidden_size

	if dual_handling == "concatenated":
		hidden_size = 2*hidden_size

	model = RegModel(plm, hidden_size, 1, 0.0, pool_method = pool_method, dual_handling = dual_handling)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	model.to(device)

	loss_fn = nn.MSELoss()

	n_epochs = 100

	total_train_steps = 0 # number of optimizer calls 
	total_steps = 0 # total passes

	MAX_TRAIN_STEPS = 10000
	EVAL_TRAIN_STEPS = 100

	results = {}

	if "prott5" in model_key:
		for layer in model.plm.encoder.block[:-1*(LAYERS_TO_TRAIN)]:
			for param in layer.parameters():
				param.requires_grad = False
	else:
		for layer in model.plm.encoder.layer[:-1*(LAYERS_TO_TRAIN)]:
			for param in layer.parameters():
				param.requires_grad = False
	
	lr = 1e-5
	weight_decay = 0.01

	optimizer = torch.optim.AdamW(
		[p for p in model.parameters() if p.requires_grad],
		lr=lr, 
		weight_decay=weight_decay
	)


	running_loss = 0.0 # dont reset at epoch, reset at train steps 
	
	

	for epoch in range(n_epochs):
		t1 = time.time()
		model.train()
		tz = time.time()
		for batch in train_loader:

			total_steps += 1
			
			lengths_1 = list(batch.pop("lengths_1")) # dont pass this to the model
			lengths_2 = list(batch.pop("lengths_2")) # dont pass this to the model

			identifiers = list(batch.pop("identifiers")) # dont pass this to the model
			labels = batch.pop("labels").to(device)

			# batch contains dict with input_ids and attention_mask

			# move only tensor entries to device
			batch = {k: v.to(device) for k, v in batch.items()}

			
			output = model(batch)


			loss = loss_fn(output, labels)

			loss.backward()
			running_loss += loss.item()


			if total_steps % gradient_accumulation_steps == 0 and total_steps > 0:
				optimizer.step()
				optimizer.zero_grad()

				total_train_steps += 1

				print(total_train_steps)
		

			if ((total_train_steps % EVAL_TRAIN_STEPS == 0) and (total_train_steps > 0)) or total_train_steps == MAX_TRAIN_STEPS:
				
				total_train_steps += 1
				model.eval()
				t2 = time.time()
				print("elapsed", total_train_steps,  round(t2 -t1, 4), "running loss", running_loss)
				
				with torch.no_grad():
					all_identities = []
					all_true_labels = []
					all_predicted_labels = []
					all_true_prob_predictions = []
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
						"predicted_label" : all_predicted_labels,
						
					}

					validataion_results = {"loss" : running_loss, "results" : val_json}


				

				with torch.no_grad():
					all_identities = []
					all_true_labels = []
					all_predicted_labels = []
					all_true_prob_predictions = []
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
						"predicted_label" : all_predicted_labels,
						
					}

					test_results = {"loss" : running_loss, "results" : test_json}


					results[total_train_steps] = {"validation" : validataion_results, "test" : test_results}

					t1 = time.time()

					print("saving ", ckpt_fn)

					with open(ckpt_fn, "w") as f1:
						json.dump(results, f1, cls=NumpyEncoder)
					f1.close()



				model.train()
				running_loss = 0.0


			if total_train_steps == MAX_TRAIN_STEPS:
				break

		if total_train_steps == MAX_TRAIN_STEPS:
			break

	with open(out_fn, "w") as f1:
		json.dump(results, f1, cls=NumpyEncoder)
	f1.close()

	print("done", out_fn)



		

