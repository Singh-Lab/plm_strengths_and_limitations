import sys 
import pandas as pd 

from config import home_path, llm_review_path
sys.path.insert(1, home_path) 
from common_functions import (
    smart_glob,
    read_fasta
)
import glob 
import os 
import pickle 
import numpy as np
import scipy.spatial  



def get_distance(emb_1, emb_2, distance_method = "mean"):
    if distance_method == "mean":
        emb_1 = emb_1[1:-1]
        v1 = np.mean(emb_1, axis = 0)
        emb_2 = emb_2[1:-1]
        v2 = np.mean(emb_2, axis = 0)
        d = scipy.spatial.distance.cosine(v1, v2)
        return d

    elif distance_method == "CLS":
        v1 = emb_1[0]
        v2 = emb_2[0]
        d = scipy.spatial.distance.cosine(v1, v2)
        return d



def depickle_obj(fn):
    with open(fn, "rb") as f:
        data = pickle.load(f)
    f.close()
    return data

if __name__ == "__main__":
    model = sys.argv[1]
    distance_method = sys.argv[2]

    refernce_files = []
    comp_df = pd.read_csv("distance_comps_reference_file.csv")

    names, seqs = read_fasta("randomly_chosen_proteins_UR50.fasta", return_headers = True)
    names = list(map(lambda x : x.split(" ")[0], names))
    name_to_seq = dict(zip(names, seqs))
    p1 = os.getcwd()  + "/extracted_embeddings/" + model + "/*.pkl"
    p2 =  os.getcwd()  + "/extracted_embeddings/" + model + "/"

    res_files = glob.glob(p1)

    reults = {}
    for i, fn in enumerate(res_files):
        name = fn.replace(p2, "").replace(".pkl", "")
        seq = name_to_seq[name]
        depickled_seq_1 = depickle_obj(fn)
        res_dict = {}
        for index, row in comp_df.iterrows():
            p_name = row["protein_name"]
            p3 = row["path"].replace("MODEL", model)
            depickled_seq_2 = depickle_obj(p3)
            d = get_distance(depickled_seq_1, depickled_seq_2, distance_method=distance_method)
            res_dict[p_name] = d
        
        reults[name] = res_dict

        if i % 100 == 0:
            print(i, model, distance_method)


    out_path = "./all_distance_results/" + model + "__" + distance_method + ".csv"
    dfo = pd.DataFrame.from_dict(reults,orient='index')
    dfo.to_csv(out_path)