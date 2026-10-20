# 
# Much of the following comes from Brandes et al.
# https://www.nature.com/articles/s41588-023-01465-0
#
# Most importantly the implementation of the weighted 
# averages of variant impact scores across tiles
# 
# Original code from https://github.com/ntranoslab/esm-variants/blob/main/esm_variants_utils.py
#

import pandas as pd 
pd.options.mode.chained_assignment = None  # default='warn'
import sys
import numpy as np
import os
import glob
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import seaborn as sns 
import scipy.stats
import random
import json
from config import llm_review_path, home_path, clinvar_file_path
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
from sklearn import metrics


def augment_masked_df_with_site_entropy(df, limit_normal_aas = True, return_entropy = True):
    df["name"] = df["ref_aa"] + df["aa_pos"].astype(str) + df["token_str"]
    df = df.loc[df["token_str"].isin(regular_aas)]
    variant_to_s1 = {}
    variant_to_adj_prob = {}
    variant_to_site_entropy = {}
    variant_to_wt_prob = {}

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
            variant_to_wt_prob[row["name"]] = wt_score
            adj_prob_list.append(row["adj_score"] / valid_prob_total)
        ent_val = entropy(adj_prob_list)
        
        for name in sub_df["name"].values: variant_to_site_entropy[name] = ent_val

    if return_entropy:
        df["site_entropy"] = df["name"].apply(lambda x : variant_to_site_entropy[x])
            
    df["s1"] = df["name"].apply(lambda x : variant_to_s1[x])
    df["adj_score"] = df["name"].apply(lambda x : variant_to_adj_prob[x])
    df["ref_score"] = df["name"].apply(lambda x : variant_to_wt_prob[x])
    df = df.set_index("name")
    return df

# Direct quote from: https://www.nature.com/articles/s41588-023-01465-0#Sec22
# Each protein sequence was tiled by iteratively generating 1,022 
# amino acid window from both ends of the sequence such that 
# consecutive windows had exactly 511 amino acid overlap until 
# windows from both ends met at the center. If the overlap between 
# the central windows was less than 511 amino acids, an additional 
# 1,022 amino acid window was added at the center 
    
    
# from : https://github.com/ntranoslab/esm-variants/blob/main/esm_variants_utils.py
    
def chop(L,min_overlap=511,max_len=1022):
    return L[max_len-min_overlap:-max_len+min_overlap]

def intervals(L,min_overlap=511,max_len=1022,parts=None):
    if parts is None: parts = []
    if len(L)<=max_len:
        if parts[-2][-1]-parts[-1][0]<min_overlap:
            #print('DIFF:',parts[-2][-1]-parts[-1][0])
            return parts+[np.arange(L[int(len(L)/2)]-int(max_len/2),L[int(len(L)/2)]+int(max_len/2)) ]
        else:
            return parts
    else:
        parts+=[L[:max_len],L[-max_len:]]
        L=chop(L,min_overlap,max_len)
        return intervals(L,min_overlap,max_len,parts=parts)

def get_intervals_and_weights(seq_len, min_overlap=511,max_len=1022,s=16):
    ints=intervals(np.arange(seq_len),min_overlap=min_overlap,max_len=max_len)
    ## sort intervals
    ints = [ints[i] for i in np.argsort([i[0] for i in ints])]

    a=int(np.round(min_overlap/2))
    t=np.arange(max_len)

    f=np.ones(max_len)
    f[:a] = 1 / (1 + np.exp(-(t[:a]-a/2)/s))
    f[max_len-a:] = 1 / (1 + np.exp((t[:a]-a/2)/s))

    f0=np.ones(max_len)
    f0[max_len-a:] = 1 / (1 + np.exp((t[:a]-a/2)/s))

    fn=np.ones(max_len)
    fn[:a] = 1 / (1 + np.exp(-(t[:a]-a/2)/s))

    filt=[f0]+[f for i in ints[1:-1]]+[fn]
    M = np.zeros((len(ints),seq_len))
    for k,i in enumerate(ints):
        M[k,i] = filt[k]
    M_norm = M/M.sum(0)
    return (ints, M, M_norm)


def get_results_for_gene(gene_name):
    brandes_subset = brandes_file_list.loc[brandes_file_list["gene"] == gene_name]
    gene_info = human_info.loc[human_info["Entry Name"] == gene_name].iloc[0]

    sequence = [*gene_info["Sequence"]]
    interval_list, weights, weights_norm = get_intervals_and_weights(len(sequence))

    brandes_subset = brandes_subset.sort_values(by = "tile_number")

    variant_to_s1_scores = {}
    variant_to_weights = {}
    variant_to_site_entropies = {}
    variant_to_alt_probs = {}
    variant_to_ref_probs = {}
    corrected_pos_to_ref = {}

    for index, row in brandes_subset.iterrows():
        tile_index = row["tile_number"]
        begin_pos = row["begin_pos"]
        end_pos = row["end_pos"]
        name = row["identifier"]
        full_tile_weights = weights_norm[tile_index]
        sequence_subset_tile_weights = full_tile_weights[begin_pos : end_pos + 1]
        print(results_path + model + "/"  + row["identifier"] + ".csv.gz")
        mlm_df = pd.read_csv(results_path + model + "/"  + row["identifier"] + ".csv.gz")
        mlm_df  = augment_masked_df_with_site_entropy(mlm_df)
        

        # all of the files are numbered from 1-L (Length), correcting for the actual position and name 
        mlm_df["corrected_position"] = mlm_df["aa_pos"].apply(lambda x : x + begin_pos)
        mlm_df["corrected_name"] = mlm_df["ref_aa"] +  mlm_df["corrected_position"].astype(str) + mlm_df["token_str"]

        for index, row in mlm_df.iterrows():
            if row["corrected_position"] in corrected_pos_to_ref.keys():
                assert row["ref_aa"] == corrected_pos_to_ref[row["corrected_position"]]
            else:
                corrected_pos_to_ref[row["corrected_position"]] = row["ref_aa"]
        
        # pos here is 1-indexed 
        for pos, mlm_sub_df in mlm_df.groupby("aa_pos"):
            site_entropy = mlm_sub_df.iloc[0]["site_entropy"]
            tile_weight = sequence_subset_tile_weights[pos - 1] # corrections from the 1-index to the 0-index 
            v_to_score = dict(zip(mlm_sub_df["corrected_name"], mlm_sub_df["s1"] ))
            v_to_prob_alt = dict(zip(mlm_sub_df["corrected_name"], mlm_sub_df["adj_score"]))
            v_to_prob_ref = dict(zip(mlm_sub_df["corrected_name"], mlm_sub_df["ref_score"]))
            for variant, s1_score in v_to_score.items():
                if variant in variant_to_s1_scores.keys():
                    variant_to_s1_scores[variant].append(s1_score)
                    variant_to_weights[variant].append(tile_weight)
                    variant_to_site_entropies[variant].append(site_entropy)
                    variant_to_ref_probs[variant].append(v_to_prob_ref[variant])
                    variant_to_alt_probs[variant].append(v_to_prob_alt[variant])
                else:
                    variant_to_s1_scores[variant] = [s1_score]
                    variant_to_weights[variant] = [tile_weight]
                    variant_to_site_entropies[variant] = [site_entropy]
                    variant_to_ref_probs[variant] = [v_to_prob_ref[variant]]
                    variant_to_alt_probs[variant] = [v_to_prob_alt[variant]]

            
    recovered_full_sequence = []
    for k, v in sort_dict_by_key(corrected_pos_to_ref).items():
        recovered_full_sequence.append(v)
        
    recovered_full_sequence = "".join(recovered_full_sequence)
    assert recovered_full_sequence == gene_info["Sequence"]

    results = []

    for k, v in variant_to_weights.items():
        pos = int(k[1:-1])
        ref = k[0]
        alt = k[-1]
        max_weight = np.max(v)
        scores = variant_to_s1_scores[k]
        
        score_range = np.max(scores) - np.min(scores)
        
        weighted_score = 0
        for wi, si in zip(v, scores):
            weighted_score += wi*si
        
        weights_str = "|".join(list(map(lambda x : str(x), v)))
        scores_str = "|".join(list(map(lambda x : str(x), variant_to_s1_scores[k])))
        entropies_str = "|".join(list(map(lambda x : str(x), variant_to_site_entropies[k])))
        ref_prob_str = "|".join(list(map(lambda x : str(x), variant_to_ref_probs[k])))
        alt_prob_str = "|".join(list(map(lambda x : str(x), variant_to_alt_probs[k])))
        results.append([
            k,
            pos,
            ref,
            alt,
            weighted_score,
            score_range,
            scores_str,
            weights_str,
            entropies_str,
            ref_prob_str,
            alt_prob_str,
            max_weight
        ])
        
    res_df = pd.DataFrame(results, columns = ["name", "aa_pos", "ref_aa", "alt_aa", "weighted_score", "score_range", "scores", "weights", "entropies", "ref_probs", "alt_probs", "max_weight"])
    return res_df

if __name__ == "__main__":
    global regular_aas
    global results_path
    global brandes_file
    global human_info
    global model

    regular_aas = ['P','S','L','A','V','G','T','E','R','Q','D','F','I','H','Y','N','K','C','W','M']
    results_path = f"{llm_review_path}masked_results/tiled_long_seqs_revised/"
    brandes_file_list = pd.read_csv(
        f"{llm_review_path}masking_pipeline_human_proteome/brandes_et_al_tiled_sequences_long_human_proteins.csv"
    )
    brandes_file_list = brandes_file_list.rename(columns = {"tile_numnber" : "tile_number"})
    human_info = pd.read_csv(f"{llm_review_path}data/uniprotkb_taxonomy_id_9606_AND_reviewed_2023_10_18.tsv", delimiter = "\t")
    model = sys.argv[1]

    output_folder = f"{llm_review_path}masked_results/combined_long_seqs_human_proteome/"
    out_dir = output_folder + model + "/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    reg_aa_set = set(regular_aas)

    gene_set =  list(set(brandes_file_list["gene"]))

    random.shuffle(gene_set) # makes multiple runs at once work

    total_num_needed = len(gene_set)

    counter = 0
    for gene_name in gene_set:
        out_fn = out_dir + gene_name + ".csv.gz"
        if os.path.exists(out_fn):
            print("skipping - already done", gene_name)
            counter += 1
            continue 

        print("doing ", gene_name, counter, "/", total_num_needed)
        gene_row = human_info.loc[human_info["Entry Name"] == gene_name].iloc[0]
        seq = gene_row["Sequence"]
        is_valid_seq = True
        observed_aas = set([*seq])

        for aa in observed_aas:
            if aa not in reg_aa_set:
                is_valid_seq = False

        if not is_valid_seq:
            print("skipping...", gene_name)
            continue 
        
        res_df = get_results_for_gene(gene_name)
        res_df.to_csv(out_dir + gene_name + ".csv.gz")
        counter += 1

    # res_df = get_results_for_gene("SYNE2_HUMAN")
