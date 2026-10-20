import numpy as np
import pandas as pd
import sys
import copy
import pickle
import time 
import os
import json 

from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import huggingface_cache_dir, results_path, progen_model_dir, home_path


sys.path.insert(1, home_path + "progen2")
from progen.progen2.models.progen.modeling_progen import ProGenForCausalLM




class LinearHead(nn.Module):
	def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.0, pool_method = "cls_pooled"):
		super().__init__()
		self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
		self.out_proj = nn.Linear(hidden_size, num_labels)
		self.pool_method = pool_method

	def forward(self, features, attention_mask):

		# features are [batch, length with padding , hidden]
		
		if features.dim() == 3:
			if self.pool_method == "cls_pooled":
				features = features[:, 0, :]   # taking CLS token gives [batch, hidden]

			elif self.pool_method == "mean_pooled":
				if attention_mask is None:
					raise ValueError("attention_mask required for mean pooling")

				# expand mask to match hidden size
				mask = attention_mask.unsqueeze(-1).clone() # [batch, seq_len, 1]
				mask = mask.to(features.dtype) #

				# --- remove CLS --- 
				
				mask[:, 0] = 0		

				# zero out padding/special tokens
				masked_features = features * mask

				# sum over sequence
				summed = masked_features.sum(dim=1)  # [batch, hidden]

				# count valid tokens
				lengths = mask.sum(dim=1)  # [batch, 1], these will match the actual seqeunce lenghts 

				# avoid division by zero
				valid_counts = mask.sum(dim=1).clamp(min=1e-9)  # [batch, 1]
				features = summed / valid_counts  # [batch, hidden]
			

		x = self.dropout(features) 
		x = x.float() # one progen uses half, other uses full precision
		x = self.out_proj(x) # [batch, hidden] -> [batch, num classes] 

		return x


class CLFModel(nn.Module):
	def __init__(
		self, 
		plm,
		hidden_size, 
		num_labels, 
		dropout, 
		pool_method = "cls_pooled"
	):
		super().__init__()
		self.plm = plm
		self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
		self.num_labels = num_labels
		self.pool_method = pool_method
		self.linear_head = LinearHead(hidden_size, num_labels, dropout, pool_method)


	def forward(self, x):
		attention_mask = x["attention_mask"]
		input_ids = x["input_ids"]
		rep = self.plm(input_ids, output_hidden_states = True).hidden_states[-1]
		x = self.linear_head(rep, attention_mask)
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

class SeqClsDataset(Dataset):
	def __init__(self, df):
		self.seqs = df["sequence"].tolist()
		self.labels = df["label"].tolist()
		self.identifiers = df["id"].tolist()
		self.lengths = list(map(lambda x : len(x), self.seqs))

	def __len__(self):
		return len(self.seqs)

	def __getitem__(self, idx):
		return {"sequence": self.seqs[idx], "labels": int(self.labels[idx]), "identifiers" : self.identifiers[idx], "lengths" : self.lengths[idx]}




def encode_batch_custom(tokenizer, seq_list, pad_token_id=0, device="cuda"):
	encodings = tokenizer.encode_batch(seq_list)
	
	ids_list = [enc.ids for enc in encodings]

	max_len = max(len(ids) for ids in ids_list)

	padded_ids = [
		ids + [pad_token_id] * (max_len - len(ids))
		for ids in ids_list
	]
	attention_mask = [
		[1] * len(ids) + [0] * (max_len - len(ids))
		for ids in ids_list
	]

	input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
	attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

	return {"input_ids": input_ids, "attention_mask": attention_mask}



def collate_fn(examples):
	sequences = ["1" + ex["sequence"] for ex in examples] # adding the 1 acting as CLS 
	labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
	identifiers = [ex["identifiers"] for ex in examples]  # keep as python list
	lengths = [ex["lengths"] for ex in examples]

	batch = encode_batch_custom(tokenizer, sequences)
	
	batch["labels"] = labels
	batch["identifiers"] = identifiers
	batch["lengths"] = lengths
	
	return batch

# from the progen files
def create_tokenizer_custom(file):
	with open(file, 'r') as f:
		return Tokenizer.from_str(f.read())





if __name__ == "__main__":
	global max_seq_len
	global model_key
	global tokenizer

	max_seq_len = 1022

	model_key = sys.argv[1]
	pool_method = sys.argv[2]
	random_state = int(sys.argv[3])

	prefix = "mar31"

	out_fn = f"{results_path}scl_new/{prefix}_{model_key}_{pool_method}_random_state_{random_state}.json"
	ckpt_fn = f"{results_path}scl_new/ckpt_{prefix}_{model_key}_{pool_method}_random_state_{random_state}.json"

	device  = "cuda" if torch.cuda.is_available() else "cpu"


	key_to_model_string = {
		"progen2_medium" : f"{progen_model_dir}progen2-medium",
		"progen2_large" : f"{progen_model_dir}progen2-large",
		"progen2_bfd" : f"{progen_model_dir}progen2-BFD90",
	}

	emb_size_dict = {
		"progen2_medium" : 1536,
		"progen2_large" : 2560,
		"progen2_bfd" : 2560
	}

	torch.manual_seed(random_state)  
	data = pd.read_csv("../../review_fine_tuning/data/scl_new_data.csv")

	data["length"] = data["sequence"].apply(lambda x : len(x))
	data = data.loc[data["length"] <= 1022]

	print("max length", np.max(data["length"].values))

	train_data = data.loc[data["partition"] == "train"] 
	validation_data = data.loc[data["partition"] == "val"] 
	test_data = data.loc[data["partition"] == "test"] 

	print("train :" , len(train_data), "val", len(validation_data), "test", len(test_data))

	model_name = key_to_model_string[model_key]

	plm = ProGenForCausalLM.from_pretrained(model_name, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

	tokenizer_fn  = f"{progen_model_dir}tokenizer.json"
	tokenizer = create_tokenizer_custom(file=tokenizer_fn)

	train_ds = SeqClsDataset(train_data)
	val_ds   = SeqClsDataset(validation_data)
	test_ds  = SeqClsDataset(test_data)	


	eff_batch_size = 64
	
	train_bs = 8
	eval_bs = 8

	if model_key == "progen2_large" or model_key == "progen2_bfd":
		train_bs = 4
		eval_bs = 4


	train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=False,  collate_fn = collate_fn)
	val_loader   = DataLoader(val_ds,   batch_size=eval_bs,  shuffle=False, collate_fn = collate_fn)
	test_loader  = DataLoader(test_ds,  batch_size=eval_bs,  shuffle=False, collate_fn = collate_fn)

	model = CLFModel(plm, emb_size_dict[model_key], 10, 0.0, pool_method = pool_method)


	LAYERS_TO_TRAIN = 4


	
	# freeze everything in the backbone first
	for param in model.plm.parameters():
		param.requires_grad = False

	# unfreeze and cast the last N transformer blocks to float32
	for layer in model.plm.transformer.h[-LAYERS_TO_TRAIN:]:
		layer.float()
		for param in layer.parameters():
			param.requires_grad = True

	# also unfreeze/fp32 the final classifier head
	model.linear_head.float()
	for param in model.linear_head.parameters():
		param.requires_grad = True


	gradient_accumulation_steps = eff_batch_size // train_bs

	

	device = "cuda" if torch.cuda.is_available() else "cpu"
	model.to(device)

	loss_fn = nn.CrossEntropyLoss()
	n_epochs = 100

	total_train_steps = 0 # number of optimizer calls 
	total_steps = 0 # total passes

	MAX_TRAIN_STEPS = 10000
	EVAL_TRAIN_STEPS = 100

	lr = 1e-5

	weight_decay = 0.01

	optimizer = torch.optim.AdamW(
		[p for p in model.parameters() if p.requires_grad],
		lr=lr, 
		weight_decay=weight_decay
	)



	results = {}
	running_loss = 0.0 # dont reset at epoch, reset at train steps 


	print("starting")
	
	for epoch in range(n_epochs):
		t1 = time.time()

		model.train()
		tz = time.time()
		for batch in train_loader:
			
			total_steps += 1
			
			lengths = list(batch.pop("lengths")) # dont pass this to the model

			identifiers = list(batch.pop("identifiers")) # dont pass this to the model
			labels = batch.pop("labels").to(device)
			# batch contains dict with input_ids and attention_mask
			# move only tensor entries to device

			batch = {k: v.to(device) for k, v in batch.items()}

			output = model(batch)

			loss = loss_fn(output, labels) / gradient_accumulation_steps
			loss.backward()

			
			running_loss += loss.item()

			if total_steps % gradient_accumulation_steps == 0 and total_steps > 0:
				# gradient clipping, try if the loss nan persists with large progen model 
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

				optimizer.step()
				optimizer.zero_grad(set_to_none=True)

				total_train_steps += 1

				print(total_train_steps)




			if ((total_train_steps % EVAL_TRAIN_STEPS == 0) and (total_train_steps > 0)) or total_train_steps == MAX_TRAIN_STEPS:
				
				total_train_steps += 1
				model.eval()
				t2 = time.time()
				# print("elapsed", total_train_steps,  round(t2 -t1, 4), "running loss", running_loss)
				
				with torch.no_grad():
					all_identities = []
					all_true_labels = []
					all_predicted_labels = []
					all_true_prob_predictions = []
					running_loss = 0 
					for batch in val_loader:
						lengths = list(batch.pop("lengths")) # dont pass this to the model
						identifiers = list(batch.pop("identifiers")) # dont pass this to the model
						labels = batch.pop("labels").to(device)

						# move only tensor entries to device
						batch = {k: v.to(device) for k, v in batch.items()}

						optimizer.zero_grad()
						output = model(batch)

						loss = loss_fn(output, labels)
						running_loss += loss.item()


						preds = torch.argmax(output, dim = 1)
						probs = torch.softmax(output, dim = 1)

						prob_true_class = probs.gather(
							dim=1,
							index=labels.unsqueeze(1)
						).squeeze(1)  


						all_identities.extend(list(identifiers))  # if idents are strings/ints/etc.
						all_true_labels.extend(labels.detach().cpu().tolist())
						all_predicted_labels.extend(preds.detach().cpu().tolist())
						all_true_prob_predictions.extend(prob_true_class.detach().cpu().tolist())



					val_json = {
						"identifier" : all_identities,
						"true_label" : all_true_labels,
						"predicted_label" : all_predicted_labels,
						"true_class_probability" : all_true_prob_predictions
					}

					validataion_results = {"loss" : running_loss, "results" : val_json}

				with torch.no_grad():
					all_identities = []
					all_true_labels = []
					all_predicted_labels = []
					all_true_prob_predictions = []
					running_loss = 0 
					for batch in test_loader:
						lengths = list(batch.pop("lengths")) # dont pass this to the model
						identifiers = list(batch.pop("identifiers")) # dont pass this to the model
						labels = batch.pop("labels").to(device)

						# move only tensor entries to device
						batch = {k: v.to(device) for k, v in batch.items()}

						optimizer.zero_grad()
						output = model(batch)

						loss = loss_fn(output, labels)
						running_loss += loss.item()

						
						preds = torch.argmax(output, dim = 1)
						probs = torch.softmax(output, dim = 1)

						prob_true_class = probs.gather(
							dim=1,
							index=labels.unsqueeze(1)
						).squeeze(1)  


						all_identities.extend(list(identifiers))  # if idents are strings/ints/etc.
						all_true_labels.extend(labels.detach().cpu().tolist())
						all_predicted_labels.extend(preds.detach().cpu().tolist())
						all_true_prob_predictions.extend(prob_true_class.detach().cpu().tolist())



					test_json = {
						"identifier" : all_identities,
						"true_label" : all_true_labels,
						"predicted_label" : all_predicted_labels,
						"true_class_probability" : all_true_prob_predictions
					}

					test_results = {"loss" : running_loss, "results" : test_json}


					results[total_train_steps] = {"validation" : validataion_results, "test" : test_results}

					t1 = time.time()

					print("saving ckpt ", ckpt_fn)

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
