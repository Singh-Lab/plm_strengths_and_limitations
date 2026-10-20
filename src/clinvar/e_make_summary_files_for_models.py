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
    smart_glob
)



if __name__ == "__main__" : 
    global model 
    model = sys.argv[1] 
    # ended up not limiting weights, limit_weights = Fasle
    # results were not meaningfully different and its best to
    # be less restrictive in the variants considered 

    limit_weights = False
    max_weight = 0.75
    min_weight = 0.25

    cols_keeping = [
        "gene_name",
        'name', 
        'aa_pos', 
        'ref_aa', 
        'alt_aa',
        "weighted_score",
        "min_weight",
        "max_weight",
        "min_score",
        "max_score",
        "min_ref_prob",
        "max_ref_prob",
        "min_entropy",
        "max_entropy",
        "num_tiles"
    ]

    combined_tile_path = f"{llm_review_path}masked_results/combined_long_seqs_human_proteome/" + model + "/"
    file_list, name_list = smart_glob(combined_tile_path)
    counter = 0 
    combined_df_parts = []
    for fn, gene_name in zip(file_list, name_list):
        df = pd.read_csv(fn)
        max_aa_pos = np.max(df["aa_pos"])
        counter += 1

        if counter % 25 == 0 :
            print("counter ", counter)
        gene_disc_counter = 0 
        gene_valid_counter = 0
        if limit_weights:
            dual_df = df.loc[(df["max_weight"] >= min_weight) & (df["max_weight"] <= max_weight)]
        else:
            dual_df = df

        dual_df["weights"] = dual_df["weights"].apply(lambda x : str(x))
        dual_df["scores"] = dual_df["scores"].apply(lambda x : str(x))
        dual_df["entropies"] = dual_df["entropies"].apply(lambda x : str(x))
        dual_df["ref_probs"] = dual_df["ref_probs"].apply(lambda x : str(x))
        dual_df["alt_probs"] = dual_df["alt_probs"].apply(lambda x : str(x))

        dual_df["min_weight"] = dual_df["weights"].apply(
            lambda x : np.min(list(map(lambda y : float(y), x.split("|"))))
        )

        dual_df["max_weight"] = dual_df["weights"].apply(
            lambda x : np.max(list(map(lambda y : float(y), x.split("|"))))
        )

        dual_df["max_score"] = dual_df["scores"].apply(
            lambda x : np.max(list(map(lambda y : float(y), x.split("|"))))
        )

        dual_df["min_score"] = dual_df["scores"].apply(
            lambda x : np.min(list(map(lambda y : float(y), x.split("|"))))
        )

        dual_df["max_entropy"] = dual_df["entropies"].apply(
            lambda x : np.max(list(map(lambda y : float(y), x.split("|"))))
        )

        dual_df["min_entropy"] = dual_df["entropies"].apply(
            lambda x : np.min(list(map(lambda y : float(y), x.split("|"))))
        )

        dual_df["max_ref_prob"] = dual_df["ref_probs"].apply(
            lambda x : np.max(list(map(lambda y : float(y), x.split("|"))))
        )

        dual_df["min_ref_prob"] = dual_df["ref_probs"].apply(
            lambda x : np.min(list(map(lambda y : float(y), x.split("|"))))
        )

        dual_df["max_ref_prob"] = dual_df["ref_probs"].apply(
            lambda x : np.max(list(map(lambda y : float(y), x.split("|"))))
        )

        dual_df["min_alt_prob"] = dual_df["alt_probs"].apply(
            lambda x : np.min(list(map(lambda y : float(y), x.split("|"))))
        )

        dual_df["max_alt_prob"] = dual_df["alt_probs"].apply(
            lambda x : np.max(list(map(lambda y : float(y), x.split("|"))))
        )

        dual_df["gene_name"] = gene_name

        dual_df["num_tiles"] = dual_df["weights"].apply(
            lambda x : len(x.split("|"))
        )

        dual_df = dual_df[cols_keeping]

        combined_df_parts.append(dual_df)
   
    if limit_weights:
        out_path_clinvar = "./combined_tiled_scores_all_genes/combined_scores_" + model + "_" + str(min_weight) + "_" + str(max_weight) +  "_clinvar_only.csv.gz"
        out_path = "./combined_tiled_scores_all_genes/combined_scores_" + model + "_" + str(min_weight) + "_" + str(max_weight) +  "_all_variants.csv.gz"
    else:
        out_path_clinvar = "./combined_tiled_scores_all_genes/combined_scores_" + model + "_all_values_clinvar_only.csv.gz"
        out_path = "./combined_tiled_scores_all_genes/combined_scores_" + model + "_all_values.csv.gz"
    
    out_df = pd.concat(combined_df_parts)

    dual_tiled_long_clinvar = []
    counter = 0 

    clinvar_long = pd.read_csv(f"{home_path}clinvar/nlp_review_misclassification/clinvar_annotations_long_genes.csv")
    clinvar_long["full_name"] = clinvar_long["entry"] + "_" + clinvar_long["name"] 
    clinvar_long_variant_to_annotation = dict(zip(clinvar_long["full_name"].values, clinvar_long["classification"].values))
    
    print(len(clinvar_long))
    print(len(out_df))
    for index, row in out_df.iterrows():
        full_name = row["gene_name"] + "_" + row["name"]
        if full_name in clinvar_long_variant_to_annotation.keys():
            dual_tiled_long_clinvar.append(clinvar_long_variant_to_annotation[full_name])
        else:
            dual_tiled_long_clinvar.append(None)
            
        if counter % 1000000 == 0:
            print("annotating... counter ", counter )
        counter += 1
    
    out_df["clinvar_annotation"] = dual_tiled_long_clinvar
    out_df.to_csv(out_path)

    out_df_2 = out_df.loc[~out_df["clinvar_annotation"].isna()]
    out_df_2.to_csv(out_path_clinvar)


