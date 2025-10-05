import json 
import pandas as pd 
import os 
import glob 
from b_make_needed_files_and_homolog_jobs import read_fasta


def get_hsp_df(json_fn):
    with open(json_fn, "r") as f2:
        data = json.load(f2)
    f2.close()

    query = data["BlastOutput2"][0]
    hits = query["report"]["results"]["search"]["hits"]
    qlen = query["report"]["results"]["search"]["query_len"]

    all_hsps = []
    for hit in hits:
        hsps = hit["hsps"]
        for hsp in hsps:
            all_hsps.append(hsp)

    hsp_df = pd.DataFrame(all_hsps)
    hsp_df["percent_identity"] = hsp_df["identity"] / (hsp_df["align_len"] - hsp_df["gaps"])
    hsp_df["percent_gaps"] = hsp_df["gaps"] / hsp_df["align_len"]
    hsp_df["alignment_frac"] = hsp_df["align_len"] / qlen

    return hsp_df, qlen


if __name__ == "__main__":
    GAP_FRAC = 0.75
    ALN_FRAC = 0.75
    EVAL = 1e-5

    regular_aas = ['P','S','L','A','V','G','T','E','R','Q','D','F','I','H','Y','N','K','C','W','M']

    cwd = os.getcwd()
    blastp_res_files = glob.glob(cwd + "/alignment_results/*.json")
    total_results = len(blastp_res_files)

    res_df = []

    VERBOSE = True

    for counter, json_fn in enumerate(blastp_res_files):
        identifier = json_fn.replace(cwd + "/alignment_results/", "").replace(".json", "")
        try:

            hsp_df, qlen = get_hsp_df(json_fn)
        except:
            print("error with" , identifier, json_fn)
            res_df.append([identifier, "error",  "error", "error", "error"])
            continue 
            
        hsp_df_2 = hsp_df.loc[
            (hsp_df["percent_gaps"] <= GAP_FRAC) &
            (hsp_df["alignment_frac"] >= ALN_FRAC) &
            (hsp_df["evalue"] <= EVAL)
        ]

        eval_df = hsp_df.loc[
            (hsp_df["evalue"] <= EVAL)
        ]

        res_df.append([identifier, len(hsp_df_2),  len(hsp_df), len(eval_df), qlen])

        if VERBOSE:
            print(counter, "/", total_results, identifier, qlen, len(hsp_df_2))

    
    res_df = pd.DataFrame(res_df, columns = ["identifier", "homologs", "raw_homologs", "e_val_homologs", "len_sequence"])
    res_df.to_csv("random_sample_homolog_counts.csv")


