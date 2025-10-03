import time
import json 
import gzip 
import random
import copy 
import numpy as np 
import pandas as pd
import sys
from config import llm_review_path, home_path

def sample_states(df, length, k):
    seq_list = []
    df = df.sample(frac = 1)
    df = df.loc[~df["sequence"].isna()]
    df["length"] = df["sequence"].apply(lambda x : len(x) if type(x) == str else None)
    df = df.loc[~df["length"].isna()]
    df = df.loc[df["length"] > k]
    init_state_seq = df.iloc[0]["sequence"][:k]
    init_state_id = df.iloc[0]["header"]
    needed_vals = length - k
    current_state = copy.deepcopy(init_state_seq)
    current_header = copy.deepcopy(init_state_id)

    valid = True 

    for i in range(needed_vals):
        valid_sample = df.loc[
            (df["sequence"].str.contains(current_state)) 
            # (df["header"] != current_header)
        ]

        valid_sample = valid_sample.sample(frac = 1)
        identified_seq = valid_sample.iloc[0]["sequence"]
        pos = identified_seq.find(current_state)
        sampled_aa = None 
        try:
            sampled_aa = identified_seq[pos + len(current_state)]
        except:
            ## this happens if the state is at the end of the sequence
            if len(valid_sample) == 1:
                return None 
            
            valid_sample = valid_sample.tail(-1)
            for index, row in valid_sample.iterrows():
                try:
                    identified_seq = row["sequence"]
                    pos = identified_seq.find(current_state)
                    sampled_aa = identified_seq[pos + len(current_state)]
                    break
                except:
                    continue 

        if sampled_aa is None:
            return None 

        current_state = current_state[1:] + sampled_aa
        current_header = valid_sample.iloc[0]["header"]
        seq_list.append(sampled_aa)

    return init_state_seq + "".join(seq_list)


if __name__ == "__main__":
    use_preload = True 

    k = int(sys.argv[1])

    counter = 0 
    t0 = time.time()
    t1 = time.time()

    if use_preload:
        dfo = pd.read_csv("ur50_seqs.csv.gz", index_col = 0)
    else:
        results = []
        sequence_file = llm_review_path + "model_source_datasets/uniref50_042021/uniref50_042021.fasta.gz"
        with gzip.open(sequence_file,'rt') as f: 
            header = True 
            while header:
                counter += 1
                if counter % 1000000 == 0:
                    print("counter ", counter , time.time() - t1)
                    t1 = time.time()
                header = f.readline().replace("\n", "")
                # header_list.append(header)
                sequence = f.readline().replace("\n", "")

                results.append({"header"  : header, "sequence" : sequence})
                # seq_list.append(sequence)
                if header is None or header == "":
                    break 
                
                if sequence is None or sequence == "":
                    break
        f.close()
        dfo = pd.DataFrame(results)
        dfo = dfo.sample(frac = 1)
        dfo.to_csv("ur50_seqs.csv.gz")


    with open("randseq_generations.fasta", "r") as f1:
        lines = f1.readlines()
    f1.close()

    lines = list(map(lambda x : x.replace("\n", ""), lines))

    entries = np.reshape(lines, (25,2))
    randseq_sequences = entries[:, 1]
    needed_lengths = list(map(lambda x : len(x), randseq_sequences))
    sampled_sequences = []
    for length in needed_lengths:
        seq = None 
        while seq is None:
            seq = sample_states(dfo, length, k)
            if seq is None:
                print("returned None... k=" + str(k) + "length = " + str(length))
        sampled_sequences.append(seq)
       

    with open("./markov_based_seqs/markov_sequences_UR50_based_window_size_" + str(k) + ".fasta", "w") as f2:
        for i, seq1 in enumerate(sampled_sequences):
            s_len = str(len(seq1))
            seq_name = f">window_size_{k}_sequence_{i};UR50 based markov_seq;length " + s_len
            f2.write(seq_name + "\n") 
            f2.write(seq1 + "\n")
    f2.close()

    
    tn = time.time()
    print(tn - t0)
    tn2 = time.time()
    print(tn2-tn)



