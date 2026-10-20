import pandas as pd 
import pandas as pd 
pd.options.mode.chained_assignment = None  # default='warn'
import sys
import numpy as np
import os
import glob
import matplotlib.pyplot as plt 
import scipy.stats

from config import llm_review_path, home_path, clinvar_file_path
sys.path.insert(1, home_path) 

from common_functions import (
    smart_glob,
    entropy
)


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

def augment_masked_df_with_site_entropy(df, limit_normal_aas = True, return_entropy = True):
    df["name"] = df["ref_aa"] + df["aa_pos"].astype(str) + df["token_str"]
    df = df.loc[df["token_str"].isin(regular_aas)]
    variant_to_s1 = {}
    variant_to_adj_prob = {}
    variant_to_site_entropy = {}
    variant_to_wt_prob = {}
    pos_to_argmax_token = {}
    pos_to_max_prob = {}
    pos_to_argmax_prob = {}

    for pos, sub_df in df.groupby("aa_pos"):
        v_to_adj_score = {}
        valid_prob_total = np.sum(sub_df["score"])
        for index, row in sub_df.iterrows():
            v_to_adj_score[row["name"]] = row["score"] / valid_prob_total
        
        sub_df["adj_score"] = sub_df["name"].apply(lambda x : v_to_adj_score[x])

        adj_score_vals = sub_df["adj_score"].values
        tokens = sub_df["token_str"].values
        argmax_prob = np.argmax(adj_score_vals)
        
        argmax_token = tokens[argmax_prob]
        pos_to_max_prob[pos] = np.max(adj_score_vals)
        pos_to_argmax_prob[pos] = argmax_token

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
    df["argmax_prob"] = df["aa_pos"].apply(lambda x : pos_to_argmax_prob[x])
    df["max_prob"] = df["aa_pos"].apply(lambda x : pos_to_max_prob[x])
    # df["argmax_token"] = 
    df = df.set_index("name")
    return df

def get_stats_for_gene(gene_name):
    brandes_subset = brandes_file_list.loc[brandes_file_list["gene"] == gene_name]
    gene_info = human_info.loc[human_info["Entry Name"] == gene_name].iloc[0]

    sequence = [*gene_info["Sequence"]]
    interval_list, weights, weights_norm = get_intervals_and_weights(len(sequence))
    brandes_subset = brandes_subset.sort_values(by = "tile_number")
    
    results_path = f"{llm_review_path}masked_results/tiled_long_seqs_revised/"

    corrected_pos_to_ref = {}

    pos_to_observed_argmax_vals = {}
    pos_to_ref_probabilities = {}
    pos_to_observed_weights = {}
    pos_to_observed_entropies = {}
    
    for index, row in brandes_subset.iterrows():
        tile_index = row["tile_number"]
        begin_pos = row["begin_pos"]
        end_pos = row["end_pos"]
        name = row["identifier"]
        full_tile_weights = weights_norm[tile_index]
        sequence_subset_tile_weights = full_tile_weights[begin_pos : end_pos + 1]
        # print(results_path + model + "/"  + row["identifier"] + ".csv.gz")
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

        mlm_positional_df = mlm_df.drop_duplicates(subset = "corrected_position")
        for index_0, row_0 in  mlm_positional_df.iterrows():
            cor_pos = row_0["corrected_position"]
            observed_argmax = row_0["argmax_prob"]
            observed_entropy = row_0["site_entropy"]
            if cor_pos in pos_to_observed_argmax_vals.keys():
                pos_to_observed_argmax_vals[cor_pos].append(observed_argmax)
                pos_to_observed_entropies[cor_pos].append(observed_entropy)
            else:
                pos_to_observed_argmax_vals[cor_pos] = [observed_argmax]
                pos_to_observed_entropies[cor_pos] = [observed_entropy]

        for cor_pos, sub_df in mlm_df.groupby("corrected_position"):
            ref_aa = sub_df.loc[sub_df["ref_aa"] == sub_df["token_str"]].iloc[0]
            ref_prob = ref_aa["adj_score"]

            if cor_pos in pos_to_ref_probabilities.keys():
                pos_to_ref_probabilities[cor_pos].append(ref_prob)
            else:
                pos_to_ref_probabilities[cor_pos] = [ref_prob]       

        for cor_pos, sub_df in mlm_df.groupby("corrected_position"):
            ref_aa = sub_df.loc[sub_df["ref_aa"] == sub_df["token_str"]].iloc[0]
            ref_prob = ref_aa["adj_score"]

            if cor_pos in pos_to_ref_probabilities.keys():
                pos_to_ref_probabilities[cor_pos].append(ref_prob)
            else:
                pos_to_ref_probabilities[cor_pos] = [ref_prob]     

             

    
    dual_vals = 0 
    single_vals = 0

    for k, v in pos_to_observed_argmax_vals.items():
        if len(v) > 1:
            seen_tokens = set(v)
            if len(seen_tokens) > 1:
                dual_vals += 1
            else:
                single_vals += 1

    prob_diffs = []
    for k, v in pos_to_ref_probabilities.items():
        if len(v) > 1:
            prob_diff_range = np.max(v) - np.min(v)
            prob_diffs.append(prob_diff_range)

    entropy_diffs = []

    for k, v in pos_to_observed_entropies.items():
        if len(v) > 1:
            ent_diff_range = np.max(v) - np.min(v)
            entropy_diffs.append(ent_diff_range)


    res = {
        "proportion_different" : dual_vals / (single_vals + dual_vals),
        "num_tiled_positions" : (single_vals + dual_vals),
        "mean_entropy_diff" : np.mean(entropy_diffs),
        "mean_ref_diff" : np.mean(prob_diffs)
    }

    return res





if __name__ == "__main__" : 
    global model 
    global regular_aas
    global brandes_file_list
    global human_info

    model = sys.argv[1] 

    regular_aas = ['P','S','L','A','V','G','T','E','R','Q','D','F','I','H','Y','N','K','C','W','M']

    human_info = pd.read_csv(f"{llm_review_path}data/uniprotkb_taxonomy_id_9606_AND_reviewed_2023_10_18.tsv", delimiter = "\t")

    brandes_file_list = pd.read_csv("../../singh/nlp_review/masking_pipeline_human_proteome/brandes_et_al_tiled_sequences_long_human_proteins.csv")
    brandes_file_list = brandes_file_list.rename(columns = {"tile_numnber" : "tile_number"})

    res_lst = []

    print(len(set(brandes_file_list["gene"])))
    gene_set = set(brandes_file_list["gene"])
    iter_counter = 0 
    for gene_name in gene_set:
        iter_counter += 1
        try:
            st = get_stats_for_gene(gene_name)
            st["gene"] = gene_name
            st["counter"] = iter_counter
            res_lst.append(st)
            print(st)
        except Exception as e:
            print("error with ",gene_name)
            print(e)

    res_df = pd.DataFrame(res_lst)
    res_df.to_csv("./f_results_all/summary_stats_" + model + ".csv")