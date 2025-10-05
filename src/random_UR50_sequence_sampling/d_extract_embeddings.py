
import pandas as pd 
from b_make_needed_files_and_homolog_jobs import read_fasta
from transformers import (
    EsmTokenizer,
    EsmModel,
    BertTokenizer,
    BertModel
)
import sys
import os 
import copy 
import pickle

def extract_embedding_for_sequence(seq, model, tokenizer):
    wt_seq = copy.deepcopy([*seq])
    tokenized_input = tokenizer(wt_seq, return_tensors="pt", is_split_into_words=True)
    output = model(**tokenized_input, output_attentions=False)
    final_hidden_wt = output.last_hidden_state.cpu().detach().numpy()[0]
    return final_hidden_wt

if __name__ == "__main__":
    fasta_path = "randomly_chosen_proteins_UR50.fasta"
    header_list, seq_list = read_fasta(fasta_path, return_headers = True)

    seq_df = pd.DataFrame({"header" : header_list, "seq" : seq_list})
    seq_df = seq_df.sample(frac = 1) # shuffle help with parallel runs 
    
    model_key = sys.argv[1] 

    key_to_model_string = {
        "esm_1b" : "facebook/esm1b_t33_650M_UR50S",
        "esm_2" :  "facebook/esm2_t36_3B_UR50D",
        "esm_1v" : "facebook/esm1v_t33_650M_UR90S_1",
        "esm_1_UR50" : "facebook/esm1_t34_670M_UR50S",
        "esm_1_UR100" : "facebook/esm1_t34_670M_UR100",
        "esm_2_15B" : "facebook/esm2_t48_15B_UR50D",
        "esm_2_650M" : "facebook/esm2_t33_650M_UR50D",
        "esm_2_150M" : "facebook/esm2_t30_150M_UR50D",
        "esm_2_35M" :  "facebook/esm2_t12_35M_UR50D",
        "esm_2_8M" :   "facebook/esm2_t6_8M_UR50D",
        "esm_1_85M": "facebook/esm1_t12_85M_UR50S",
        "esm_1_43M": "facebook/esm1_t6_43M_UR50S",
        "protbert" : "Rostlab/prot_bert",
        "protbert_bfd" : "Rostlab/prot_bert_bfd"
    }

    model_name = key_to_model_string[model_key]

    cwd = os.getcwd()
    out_dir = cwd + "/extracted_embeddings/" + model_key + "/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    if "esm_2" in model_key:
        tokenizer = EsmTokenizer.from_pretrained(
            model_name,
            do_lower_case=False,
        )
        model = EsmModel.from_pretrained(
            model_name
        )
        
    elif "esm" in model_key:
        tokenizer = EsmTokenizer.from_pretrained(
            model_name,
            do_lower_case=False,
        
        )
        model = EsmModel.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True,
        
        )
        
    else:
        tokenizer = BertTokenizer.from_pretrained(
            model_name,
            do_lower_case=False,
            force_download=True,
        
        )
        model = BertModel.from_pretrained(
            model_name
        )

    counter = 0 
    for index, row in seq_df.iterrows():
        header = row["header"].split(" ")[0]
        seq = row["seq"]
        out_fn = out_dir + header + ".pkl"

        if os.path.exists(out_fn):
            continue 

        counter += 1
        print("doing ",header, counter )
        embedding = extract_embedding_for_sequence(seq, model, tokenizer)

        with open(out_fn, "wb") as f:
            pickle.dump(embedding, f)
        f.close()

        
 