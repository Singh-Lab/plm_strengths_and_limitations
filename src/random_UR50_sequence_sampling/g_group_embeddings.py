import pandas as pd 

import numpy as np 
import glob 
import scipy.stats
import scipy.spatial 
import pickle 
import sys

from config import home_path
sys.path.insert(1, home_path) 
from common_functions import (
    my_colors, 
    format_p, 
    get_pfam_info, 
    mean_confidence_interval, 
    sort_dict_by_key, 
    sort_dict_by_value,
    three_letter_to_one_letter,
    entropy,
    smart_glob
)
def depickle_obj(fn):
    with open(fn, "rb") as f:
        data = pickle.load(f)
    f.close()
    return data

def pickle_obj(obj, fn):
    with open(fn, "wb") as f:
        pickle.dump(obj, f)
    f.close()



if __name__ == "__main__":

    for model_of_interest in  ["esm_2_8M", "esm_2_35M", "esm_2_150M", "esm_2_650M", "esm_2", "esm_2_15B"]:
        embedding_path = "./extracted_embeddings/" + model_of_interest + "/"
        files = glob.glob(embedding_path + "*.pkl")
        sampled_embs = {"mean" : {}, "cls" : {}}
        print(model_of_interest, len(files))
        for emb_path in files:
            name = emb_path.replace(embedding_path, "").replace(".pkl", "")
            emb = depickle_obj(emb_path)
            cls_emb = emb[0]
            emb = emb[1:-1,]
            emb = np.mean(emb, axis = 0)

            sampled_embs["mean"][name] = emb
            sampled_embs["cls"][name] = cls_emb

        pickle_obj(sampled_embs, "./grouped_extrcted_embeddings/" + model_of_interest + ".pkl")
