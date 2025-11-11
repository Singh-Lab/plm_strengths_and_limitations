from transformers import (
    EsmForMaskedLM, 
    BertTokenizer, 
    EsmTokenizer,
    pipeline,
    BertForMaskedLM
)
import torch
import pandas as pd
import copy 
import sys
import time 
import random
import glob
import os

from masking_tools import load_model_and_alphabet_local, get_iterative_masked_seqeunce, augment_masked_df_with_site_entropy


def get_iterative_masked_df(model_key, sequence, sequence_name, direct_loading_model_path = None, huggingface_cache_dir = "./", device_str = None, force_direct_loading = False, verbose = False):
    key_to_model_string = {
        "esm_1b" : "facebook/esm1b_t33_650M_UR50S",
        "esm_1v" : "facebook/esm1v_t33_650M_UR90S_1",
        "esm_2_15B" : "facebook/esm2_t48_15B_UR50D",
        "esm_2_3B" :  "facebook/esm2_t36_3B_UR50D",
        "esm_2_650M" : "facebook/esm2_t33_650M_UR50D",
        "esm_2_150M" : "facebook/esm2_t30_150M_UR50D",
        "esm_2_35M" :  "facebook/esm2_t12_35M_UR50D",
        "esm_2_8M" :   "facebook/esm2_t6_8M_UR50D",
        "protbert" : "Rostlab/prot_bert"
    }

    using_direct_loading = True
    if model_key in key_to_model_string.keys():
        using_direct_loading = False

    if force_direct_loading:
        using_direct_loading = True



    use_esm = False 
    if "esm" in model_key:
        use_esm = True

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_str)
    device_integer = 0 if torch.cuda.is_available() else -1
    
    if using_direct_loading:
        model, alphabet = load_model_and_alphabet_local(direct_loading_model_path)
        model.eval()
        model.to(device)
        batch_converter = alphabet.get_batch_converter()
        df_o = get_iterative_masked_seqeunce(model, alphabet, batch_converter, sequence, "seq_1", device_str = device_str)

    else:
        model_name = key_to_model_string[model_key]
        if use_esm:
            mask_token = "<mask>"
            if "esm_2" in model_key:
                tokenizer = EsmTokenizer.from_pretrained(model_name, do_lower_case=False, cache_dir=huggingface_cache_dir)
                model = EsmForMaskedLM.from_pretrained(model_name, cache_dir=huggingface_cache_dir)
            else:
                tokenizer = EsmTokenizer.from_pretrained(model_name, do_lower_case=False, cache_dir=huggingface_cache_dir)
                model = EsmForMaskedLM.from_pretrained(model_name, ignore_mismatched_sizes=True, cache_dir=huggingface_cache_dir)
        else:    
            mask_token = "[MASK]"
            tokenizer = BertTokenizer.from_pretrained(
                model_name,
                do_lower_case=False, 
                cache_dir=huggingface_cache_dir
            )
            model = BertForMaskedLM.from_pretrained(model_name, cache_dir=huggingface_cache_dir)


        model.to(device)
        model.eval()
        unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer, top_k = tokenizer.vocab_size, device = device_integer)

        wt_sequence = copy.deepcopy(sequence)
        wt_sequence = [*sequence]
        all_results = []
        length = len(sequence)
        
        for i in range(len(sequence)):
            if verbose:
                print(sequence_name, str(i), "/", str(length))

            wt_aa = wt_sequence[i]
            masked_seqeunce = copy.deepcopy(wt_sequence)
            masked_seqeunce[i] = copy.deepcopy(mask_token) 
            if model_key in ["esm_1b", "esm_2"]:
                masked_seqeunce = "".join(masked_seqeunce)
            else:
                masked_seqeunce = " ".join(masked_seqeunce)
            res = unmasker(masked_seqeunce)
            for aa_res in res:
                _ =  aa_res.pop("sequence")
                aa_res["aa_pos"] = i + 1
                aa_res["ref_aa"] = wt_aa
                aa_res["gene"] = sequence_name
                all_results.append(aa_res)

        df_o = pd.DataFrame(all_results)
    
    res_df = augment_masked_df_with_site_entropy(df_o)
    return res_df



if __name__ == "__main__":
    model_key = "esm_2_35M"

    huggingface_cache_dir = "/scratch/gpfs/jf9645/huggingface_alt_cache/"

    
