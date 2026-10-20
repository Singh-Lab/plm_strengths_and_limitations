import argparse
import pickle
import glob
import os
import time
import json
import sys
import copy
import scipy.stats
import pandas as pd
import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from config import results_path, dataset_path, shared_path, sparse_dataset_path



class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return super().default(obj)


class MyDataset(Dataset):
	def __init__(self, X, y, identifiers):
		self.input_dimension= len(X[0])
		self.X = X
		self.y = y
		self.identifiers = identifiers

	def set_input_dimension(self, input_dimension):
		self.input_dimension = input_dimension

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx], self.identifiers[idx]


class RegModel(nn.Module):
	def __init__(
		self,
		input_dim = 1280,
		hidden_dim_factor = 0.5,
		seed = 0,
	):
		super(RegModel, self).__init__()
		torch.manual_seed(seed)
		
		self.seed = seed
		self.input_dim = input_dim
		self.hidden_dim_factor = hidden_dim_factor
		self.hidden_dim_size = int(self.input_dim * self.hidden_dim_factor)
		
		self.fc1 = nn.Linear(self.input_dim, self.hidden_dim_size)
		self.fc2 = nn.Linear(self.hidden_dim_size, 1)



	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x).squeeze(-1)  # output shape: [batch_size]
		return x


def depickle_obj(fn):
	with open(fn, "rb") as f1:
		data = pickle.load(f1)
	f1.close()

	return data

def get_dataset_dict(task, model_name, pool_method):

	task_str = task
	if task == "ppi_affinity":
		task_str = "ppi_affinity_all" # name change from previous to current setup, data is the same

	fn_1 = sparse_dataset_path + task_str + "/raw/" + model_name + "_" + pool_method + ".pkl"

	if os.path.exists(fn_1):
		

		ds_preload_path = (
			sparse_dataset_path + task_str + "/raw/" + model_name + "_" + pool_method + ".pkl"
		)
		ds = depickle_obj(ds_preload_path)

	else:
		fn_2 = f"{shared_path}review_downstream/preloads_02022026/{task}/{model_name}_{pool_method}.pkl"
		ds = depickle_obj(fn_2)
	
	return ds


if __name__ == "__main__":
	model_name = sys.argv[1]
	pool_method = sys.argv[2]

	task = "ppi_affinity" 

	prefix = "feb_27"
	seed_to_res = {}

	res_fn = results_path + task + "/" + prefix + "_" + model_name + "_" + pool_method + ".json"
	print(res_fn)

	if not os.path.exists(results_path + task):
		os.mkdir(results_path + task)

	num_seeds = 10
	batch_size = 32
	n_classes = 2
	learning_rate = 1e-5
	weight_decay = 0
	num_epochs = 100

	print("res", res_fn)
	print("dataset", f"{dataset_path}{task}/{model_name}_{pool_method}.pkl")

	dataset_dict = get_dataset_dict(task, model_name, pool_method)
	input_dimension = dataset_dict["train"].input_dimension

	print(model_name, pool_method, task, input_dimension)
	device = "cuda" if torch.cuda.is_available() else "cpu"

	print(dataset_dict["train"].X.size(), input_dimension)
	
	t0 = time.time()

	for seed in range(num_seeds):

		print("on seed", seed, model_name, pool_method, task)
		torch.manual_seed(seed)

		model = RegModel(input_dim = input_dimension, seed = seed, hidden_dim_factor = 0.5)
		model.to(device)

		loss_fn = nn.MSELoss()
		
		
		optimizer = optim.Adam(
			model.parameters(), 
			lr=learning_rate,
			weight_decay=weight_decay
		)

		train_dataloader = DataLoader(dataset_dict["train"], batch_size=batch_size, shuffle=False)
		validation_dataloader = DataLoader(dataset_dict["validation"], batch_size=batch_size, shuffle=False)
		test_dataloader = DataLoader(dataset_dict["test"], batch_size=batch_size, shuffle=False)



		model.train()
		
		epoch_results = {}

		for epoch in range(num_epochs):
			t1 = time.time()

			model.train()

			running_loss = 0.0

			for batch_X, batch_y, idents  in train_dataloader:


				batch_X = torch.mean(batch_X, dim =1) # averaging the 2 proteins in the PPI dataset 
				batch_X = batch_X.to(device)
				batch_X = batch_X.to(torch.float32)
				
				batch_y = batch_y.to(device)
				batch_y = batch_y.to(torch.float32)
				
				preds = model(batch_X)
				loss = loss_fn(preds, batch_y)
				running_loss += loss.item()
				# batch_y = torch.argmax(batch_y, dim=1).long()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()


			
			running_loss = 0.0
			model.eval()

			with torch.no_grad():

				all_identities = []
				all_true_labels = []
				all_predicted_labels = []
				

				running_loss = 0.0

				for batch_X, batch_y, idents in validation_dataloader:
					# print(type(idents), type(idents[0]))

					if type(idents[0]) == tuple:
						idents = [x1 + "_" + x2 for (x1, x2) in zip(idents[0], idents[1])]

					batch_X = torch.mean(batch_X, dim =1) # averaging the 2 proteins in the PPI dataset 
					batch_X = batch_X.to(device)
					batch_X = batch_X.to(torch.float32)
					
					batch_y = batch_y.to(device)
					batch_y = batch_y.to(torch.float32)

					preds = model(batch_X)
					loss = loss_fn(preds, batch_y)
					running_loss += loss.item()

					all_identities.extend(idents) 
					all_true_labels.extend(batch_y.detach().cpu().tolist())
					all_predicted_labels.extend(preds.detach().cpu().tolist())

				

				validation_json = {
					"identifier" : all_identities,
					"true_label" : all_true_labels,
					"predicted_label" : all_predicted_labels,
				}


				validataion_results = {"loss" : running_loss, "results" : validation_json}

				all_identities = []
				all_true_labels = []
				all_predicted_labels = []

				running_loss = 0.0

				for batch_X, batch_y, idents in test_dataloader:
					if type(idents[0]) == tuple:
						idents = [x1 + "_" + x2 for (x1, x2) in zip(idents[0], idents[1])]

					batch_X = torch.mean(batch_X, dim =1) # averaging the 2 proteins in the PPI dataset 
					batch_X = batch_X.to(device)
					batch_X = batch_X.to(torch.float32)
					
					batch_y = batch_y.to(device)
					batch_y = batch_y.to(torch.float32)

					preds = model(batch_X)
					loss = loss_fn(preds, batch_y)
					running_loss += loss.item()
					

					all_identities.extend(idents) # if idents are strings/ints/etc.
					all_true_labels.extend(batch_y.detach().cpu().tolist())
					all_predicted_labels.extend(preds.detach().cpu().tolist())



				test_json = {
					"identifier" : all_identities,
					"true_label" : all_true_labels,
					"predicted_label" : all_predicted_labels
				}

				test_results = {"loss" : running_loss, "results" : test_json}

			epoch_results[epoch] = {"validation" : validataion_results, "test" : test_results}

		seed_to_res[seed] = epoch_results
	
	t1 = time.time()

	print("finished, in ", t1 - t0, "seconds")

	with open(res_fn, "w") as f:
		json.dump(seed_to_res, f, cls=NumpyEncoder)
	f.close()

	print("done", res_fn)


