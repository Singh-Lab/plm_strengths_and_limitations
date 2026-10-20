import pandas as pd 
pd.options.mode.chained_assignment = None  # default='warn'
import sys
import numpy as np
import os
import glob
import matplotlib.pyplot as plt 
import scipy.stats
from config import llm_review_path, home_path
sys.path.insert(1, home_path) 
from common_functions import (
    my_colors, 
    format_p, 
    get_pfam_info, 
    mean_confidence_interval, 
    sort_dict_by_key, 
    sort_dict_by_value,
    three_letter_to_one_letter,
    entropy
)
import seaborn as sns 
from sklearn import metrics
import time


def augment_masked_df_with_site_entropy(df, limit_normal_aas = True, return_entropy = True):
    df["name"] = df["ref_aa"] + df["aa_pos"].astype(str) + df["token_str"]
    df = df.loc[df["token_str"].isin(regular_aas)]
    variant_to_s1 = {}
    variant_to_adj_prob = {}
    variant_to_site_entropy = {}
    variant_to_adj_ref_prob = {}

    for pos, sub_df in df.groupby("aa_pos"):
        v_to_adj_score = {}
        valid_prob_total = np.sum(sub_df["score"])
        for index, row in sub_df.iterrows():
            v_to_adj_score[row["name"]] = row["score"] / valid_prob_total
        
        sub_df["adj_score"] = sub_df["name"].apply(lambda x : v_to_adj_score[x])
        
        wt_val = sub_df.loc[sub_df["ref_aa"] == sub_df["token_str"]]
        assert len(wt_val) == 1
        wt_score = wt_val.iloc[0]["adj_score"]
        
        adj_prob_list = []
        for index, row in sub_df.iterrows():
            variant_to_s1[row["name"]] = np.log(row["adj_score"]) - np.log(wt_score)
            variant_to_adj_prob[row["name"]] = row["adj_score"] / valid_prob_total
            variant_to_adj_ref_prob[row["name"]] = wt_score
            adj_prob_list.append(row["adj_score"] / valid_prob_total)
        ent_val = entropy(adj_prob_list)
        
        for name in sub_df["name"].values: variant_to_site_entropy[name] = ent_val

    if return_entropy:
        df["site_entropy"] = df["name"].apply(lambda x : variant_to_site_entropy[x])
            
    df["s1"] = df["name"].apply(lambda x : variant_to_s1[x])
    df["adj_score"] = df["name"].apply(lambda x : variant_to_adj_prob[x])
    df["ref_score"] = df["name"].apply(lambda x : variant_to_adj_ref_prob[x]) 
    df = df.set_index("name")
    return df

if __name__ == "__main__":
    global regular_aas     
    regular_aas = ['P','S','L','A','V','G','T','E','R','Q','D','F','I','H','Y','N','K','C','W','M']

    clinvar_variants = pd.read_csv("clinvar_variants_662_short_genes.csv")

    model = sys.argv[1]

    mask_path = f"{llm_review_path}masked_results/human_proteome/" + model + "/"

    variant_to_score = {}
    variant_to_prob = {}
    variant_to_entropy = {}
    variant_to_ref_prob = {}
    needed_entries = set(clinvar_variants["mapped_gene_file"])
    print(len(needed_entries))

    t1 = time.time()
    
    auroc_scores = []
    results = []

    for i, (entry, sub_df) in enumerate(clinvar_variants.groupby("mapped_gene_file")):
        if i % 50 ==0 :
            print("counter ", i)
            
        sub_df["full_name"] = entry + "_" + sub_df["variant_name"]
        mlm_df = pd.read_csv(mask_path + entry + ".csv.gz")
        mlm_df = augment_masked_df_with_site_entropy(mlm_df)
    
            
        masked_scores = dict(zip(mlm_df.index.values, mlm_df["s1"].values))
        prob_dict = dict(zip(mlm_df.index.values, mlm_df["adj_score"].values))
        entropy_dict = dict(zip(mlm_df.index.values, mlm_df["site_entropy"].values))
        ref_prob_dict = dict(zip(mlm_df.index.values, mlm_df["ref_score"].values))

        for index, row in sub_df.iterrows():
            variant_to_score[row["full_name"]] = masked_scores[row["variant_name"]]
            variant_to_entropy[row["full_name"]] = entropy_dict[row["variant_name"]]
            variant_to_prob[row["full_name"]] = prob_dict[row["variant_name"]]
            variant_to_ref_prob[row["full_name"]] = ref_prob_dict[row["variant_name"]]
            
        plp_variants = sub_df.loc[sub_df["PLP"]]["variant_name"].values
        blb_variants = sub_df.loc[sub_df["BLB"]]["variant_name"].values
        
        variants = list(plp_variants) + list(blb_variants)

        plp_scores = [masked_scores[x] for x in plp_variants]
        blb_scores = [masked_scores[x] for x in blb_variants]
        
        labels = [1] * len(plp_scores) + [0] * len(blb_scores)
        
        predictions =  list(plp_scores) + list(blb_scores)
        
        fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, pos_label=0)
        auroc = metrics.auc(fpr, tpr)
        auroc_scores.append(auroc)

        predictions_str = "|".join(list(map(lambda x : str(round(x, 5)), predictions)))
        labels_str = "|".join(list(map(lambda x : str(x), labels)))
        variants = "|".join(variants)

        results.append([entry, len(plp_scores), len(blb_scores), variants, predictions_str, labels_str, auroc])
        
    t2 = time.time()

    dfo = pd.DataFrame(results, columns = ["gene", "num_PLP", "num_BLB", "variants", "predictions", "labels", "auroc"])
    dfo.to_csv("./model_clinvar_results/" + model + ".csv")