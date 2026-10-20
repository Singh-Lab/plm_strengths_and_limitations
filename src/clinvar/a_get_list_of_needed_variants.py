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


def extract_coding_info_from_name(name_str, return_ref = True, return_pos = False, return_alt = False):
    try:
        pieces = name_str.split("(p.")
        needed_piece = pieces[1]
        ref_aa = needed_piece[:3]
        piece_2 = needed_piece.split(")")[0]
        if piece_2[-1] == "=":
            alt_aa = ref_aa
            aa_pos = int(piece_2[3:-1])
        else:
            alt_aa = piece_2[-3:]
            aa_pos = int(piece_2[3:-3])
        if return_alt:
            return alt_aa
        elif return_pos:
            return aa_pos
        else:
            return ref_aa
    except:
        print("error", name_str)
        return None
 


def get_clinvar_df(clinvar_fn):

    human_info = pd.read_csv(f"{llm_review_path}data/uniprotkb_taxonomy_id_9606_AND_reviewed_2023_10_18.tsv", delimiter = "\t")
    entry_to_uniport_id = dict(zip(human_info["Entry Name"].values,human_info["Entry"].values))
    gene_to_entry = {}
    for index, row in human_info.iterrows():
        gene_vals = row["Gene Names"]
        if pd.isna(gene_vals):
            gene_vals = [entry_name.replace("_HUMAN", "")]
        else:
            
            gene_vals = row["Gene Names"].split(" ")
            entry_name = row["Entry Name"]
        for g in gene_vals:
            gene_to_entry[g.upper()] = entry_name

    df = pd.read_csv(clinvar_fn, delimiter = "\t")
    df = df.loc[df["Type"] == "single nucleotide variant"] # removes indels + other non SNP variants 

    my_regex = ".*c\.[0-9]*[ATCG]>[ATCG] \(p\.[A-Z][a-z]{2}[0-9]*[^_]*\)"
    df2 = df.loc[df["Name"].str.match(my_regex)] # gets the matches to the regex 
    df2["ref_aa"] = df2["Name"].apply(lambda x : extract_coding_info_from_name(x, return_ref = True,  return_pos = False, return_alt = False))
    df2["alt_aa"] = df2["Name"].apply(lambda x : extract_coding_info_from_name(x, return_ref = False, return_pos = False, return_alt = True))
    df2["aa_pos"] = df2["Name"].apply(lambda x : extract_coding_info_from_name(x, return_ref = False, return_pos = True,  return_alt = False))
    clinvar_missense = df2.loc[
        (df2["alt_aa"] != df2["ref_aa"]) &
        (df2["alt_aa"] != "Ter") &
        (df2["ref_aa"] != "Ter") &
        (df2["aa_pos"] != 1)
    ]

    clinvar_missense["ref_aa_1_letter"] = clinvar_missense["ref_aa"].apply(lambda x : three_letter_to_one_letter[x.upper()])
    clinvar_missense["alt_aa_1_letter"] = clinvar_missense["alt_aa"].apply(lambda x : three_letter_to_one_letter[x.upper()])
    clinvar_missense["variant_name"] = clinvar_missense["ref_aa_1_letter"] +clinvar_missense["aa_pos"].astype(str) + clinvar_missense["alt_aa_1_letter"]

    clinvar_missense_annotated = clinvar_missense.loc[
        clinvar_missense["ClinicalSignificance"].isin(p_lp_annotations + b_lb_annotations + vus_annotations)
    ]

    clinvar_missense_annotated["PLP"] = clinvar_missense["ClinicalSignificance"].apply(lambda x : x in p_lp_annotations)
    clinvar_missense_annotated["BLB"] = clinvar_missense["ClinicalSignificance"].apply(lambda x : x in b_lb_annotations)
    clinvar_missense_annotated["VUS"] = clinvar_missense["ClinicalSignificance"].apply(lambda x : x in vus_annotations)

    clinvar_genes = set(clinvar_missense_annotated["GeneSymbol"])

    not_found_genes = set()
    found_genes = set()

    gene_symbol_map = {}

    for g in clinvar_genes:
        gene_found = False 
        g = g.upper()
        if g in gene_to_entry.keys():
            gene_found = True  
            gene_symbol_map[g] = gene_to_entry[g]
        else:
            if ";" in g:
                pieces = g.split(";")
                for piece in pieces:
                    if piece in gene_to_entry.keys():
                        gene_found = True
                        gene_symbol_map[g] = gene_to_entry[piece]

            if "-" in g:
                pieces = g.split("-")
                for piece in pieces:
                    if piece in gene_to_entry.keys():
                        gene_found = True 
                        gene_symbol_map[g] = gene_to_entry[piece]

        if gene_found:
            found_genes.add(g)
        else:
            not_found_genes.add(g)

    clinvar_missense_annotated["mapped_gene_file"] = clinvar_missense_annotated["GeneSymbol"].apply(lambda x : gene_to_entry[x] if x in gene_to_entry.keys() else None)
    clinvar_missense_annotated["uniprot"] = clinvar_missense_annotated["mapped_gene_file"].apply(lambda x : entry_to_uniport_id[x] if x in entry_to_uniport_id.keys() else None)
    
    return clinvar_missense_annotated


if __name__ == "__main__":
  

    global p_lp_annotations
    global b_lb_annotations
    global vus_annotations

    p_lp_annotations = [ 'Pathogenic/Likely pathogenic','Likely pathogenic',  'Pathogenic']
    b_lb_annotations = ['Likely benign','Benign', 'Benign/Likely benign']
    vus_annotations = ["Uncertain significance/Uncertain risk allele", "Uncertain significance"]
    
    clinvar_df = get_clinvar_df(clinvar_file_path)
    clinvar_df = clinvar_df.loc[~clinvar_df["mapped_gene_file"].isna()]

    # using esm_2 as a guide for seeing those that match the ref seqeunce, any model will do they're all the same 
    mask_path = f"{llm_review_path}masked_results/human_proteome/esm_2/"
    masked_results_path_short_genes = glob.glob(mask_path+"*")
    masked_results_path_short_genes = list(map(lambda x : x.replace(mask_path, "").replace(".csv.gz", ""), masked_results_path_short_genes))

    clinvar_df = clinvar_df.loc[
        (~clinvar_df["mapped_gene_file"].isna())
        & (clinvar_df["mapped_gene_file"].isin(masked_results_path_short_genes))
    ]

    human_info = pd.read_csv(f"{llm_review_path}data/uniprotkb_taxonomy_id_9606_AND_reviewed_2023_10_18.tsv", delimiter = "\t")

    regular_aas = set(['P','S','L','A','V','G','T','E','R','Q','D','F','I','H','Y','N','K','C','W','M'])

    random_seed = 0 
    success_counter = 0
    failure_counter = 0  
    inclusion_counter = 0 


    class_cutoff = 5

    results = []

    for gene_entry, sub_df in clinvar_df.groupby("mapped_gene_file"):

        human_entry = human_info.loc[human_info["Entry Name"] == gene_entry].iloc[0]
        sequence = human_entry["Sequence"]
           
        aas = set([*sequence])

        # removing sequences with irregular aas 
        valid_seq = True 
        for aa in aas: 
            if aa not in regular_aas:
                valid_seq = False 

        if not valid_seq:
            print("invalid ", gene_entry)
            continue 


        # ignoring genes where reports do not match reference in our set 
        sequence_clinvar_match = True
        sequence = [*sequence]
        seq_len = len(sequence)
        sub_df = sub_df.loc[sub_df["aa_pos"] != 1]
        sub_df  = sub_df.drop_duplicates(subset = "variant_name")

      
        for index, row in sub_df.iterrows():
            reported_ref = row["ref_aa_1_letter"]
            reported_pos = row["aa_pos"]

            if reported_pos > seq_len:
                sequence_clinvar_match = False 
            else:
                recovered_ref = sequence[reported_pos - 1]
                if recovered_ref != reported_ref:
                    sequence_clinvar_match = False 


        if sequence_clinvar_match:
            success_counter += 1
        else:
            failure_counter += 1

        
        if not sequence_clinvar_match:
            continue 

        vus_df = sub_df.loc[sub_df["VUS"]]
        plp_df = sub_df.loc[sub_df["PLP"]]
        blb_df = sub_df.loc[sub_df["BLB"]]

        if (len(vus_df) >= class_cutoff) and (len(plp_df) >= class_cutoff) and (len(blb_df) >= class_cutoff):
            inclusion_counter += 1
            vus_selections = vus_df.sample(frac = 1, random_state = random_seed).head(class_cutoff)
            plp_selections = plp_df.sample(frac = 1, random_state = random_seed).head(class_cutoff)
            blb_selections = blb_df.sample(frac = 1, random_state = random_seed).head(class_cutoff)

            for index, row in vus_selections.iterrows():
                results.append([
                    row["ref_aa"],
                    row["aa_pos"],
                    row["alt_aa"],
                    row["variant_name"],
                    row["uniprot"],
                    gene_entry,
                    "VUS"
                ])

            for index, row in plp_selections.iterrows():
                results.append([
                    row["ref_aa"],
                    row["aa_pos"],
                    row["alt_aa"],
                    row["variant_name"],
                    row["uniprot"],
                    gene_entry,
                    "PLP"
                ])

            for index, row in blb_selections.iterrows():
                results.append([
                    row["ref_aa"],
                    row["aa_pos"],
                    row["alt_aa"],
                    row["variant_name"],
                    row["uniprot"],
                    gene_entry,
                    "BLB"
                ])
    
    dfo = pd.DataFrame(results, columns = ["ref", "pos", "alt", "name", "uniprot", "entry", "classification"])
    dfo.to_csv("clinvar_selections_short_genes.csv")
    print(success_counter, failure_counter, inclusion_counter)
        




        