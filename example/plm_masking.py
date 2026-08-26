from transformers import (
    EsmForMaskedLM,
    BertTokenizer,
    EsmTokenizer,
    pipeline,
    BertForMaskedLM,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
import torch
import pandas as pd
import copy
import sys
import os

from masking_tools import (
    load_model_and_alphabet_local,
    get_iterative_masked_seqeunce,
    get_iterative_masked_carp,
    get_iterative_masked_progen2,
    get_iterative_masked_prott5,
    augment_masked_df_with_site_entropy,
)

CARP_KEYS = ["carp_600k", "carp_38M", "carp_76M", "carp_640M"]
PROGEN2_KEYS = ["progen2_medium", "progen2_large", "progen2_bfd"]
PROGEN2_VARIANT = {"progen2_medium": "medium", "progen2_large": "large", "progen2_bfd": "BFD90"}
PROTT5_KEY_TO_MODEL_STRING = {
    "prott5_xl": "Rostlab/prot_t5_xl_uniref50",
    "prott5_xl_bfd": "Rostlab/prot_t5_xl_bfd",
}


def get_iterative_masked_df(model_key, sequence, sequence_name, direct_loading_model_path = None, huggingface_cache_dir = None, device_str = None, force_direct_loading = False, verbose = False, progen_model_dir = None, progen_code_dir = None, carp_model_path = None):
    if not sequence:
        raise ValueError("sequence must be non-empty")

    key_to_model_string = {
        "esm_1b" : "facebook/esm1b_t33_650M_UR50S",
        "esm_1v" : "facebook/esm1v_t33_650M_UR90S_1",
        "esm_2_15B" : "facebook/esm2_t48_15B_UR50D",
        "esm_2_3B" :  "facebook/esm2_t36_3B_UR50D",
        "esm_2_650M" : "facebook/esm2_t33_650M_UR50D",
        "esm_2_150M" : "facebook/esm2_t30_150M_UR50D",
        "esm_2_35M" :  "facebook/esm2_t12_35M_UR50D",
        "esm_2_8M" :   "facebook/esm2_t6_8M_UR50D",
        "protbert" : "Rostlab/prot_bert",
        "protbert_bfd" : "Rostlab/prot_bert_bfd"
    }

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_str)
    device_integer = (device.index or 0) if device.type == "cuda" else -1

    if model_key in CARP_KEYS:
        from sequence_models.pretrained import load_model_and_alphabet as load_carp

        # carp_model_path lets a caller point at an already-downloaded local .pt
        # (e.g. on a network-restricted compute node) instead of relying on the
        # library's automatic Zenodo download, which needs internet access.
        model, collater = load_carp(carp_model_path if carp_model_path else model_key)
        model.to(device)
        model.eval()
        df_o = get_iterative_masked_carp(model, collater, sequence, sequence_name, device_str=device_str)
        res_df = augment_masked_df_with_site_entropy(df_o)
        return res_df

    if model_key in PROGEN2_KEYS:
        if progen_model_dir is None:
            raise ValueError(
                f"{model_key} requires progen_model_dir (a local directory containing "
                f"progen2-{PROGEN2_VARIANT[model_key]}/, e.g. as downloaded from "
                "https://github.com/salesforce/progen)"
            )
        if progen_code_dir is None:
            raise ValueError(
                f"{model_key} requires progen_code_dir: the local directory containing "
                "a clone of https://github.com/salesforce/progen, i.e. the parent of "
                "the 'progen' package (progen/progen2/models/progen/modeling_progen.py)"
            )
        if "progen" in sys.modules:
            loaded_from = os.path.dirname(os.path.dirname(sys.modules["progen"].__file__ or ""))
            if os.path.abspath(loaded_from) != os.path.abspath(progen_code_dir):
                raise RuntimeError(
                    "The 'progen' package was already imported from a different "
                    f"progen_code_dir ({loaded_from}) earlier in this process; Python "
                    "caches imports by module name, so a new progen_code_dir "
                    f"({progen_code_dir}) can't take effect here. Restart Python before "
                    "using a different progen_code_dir."
                )
        if progen_code_dir not in sys.path:
            sys.path.insert(0, progen_code_dir)

        from tokenizers import Tokenizer
        from progen.progen2.models.progen.modeling_progen import ProGenForCausalLM

        model_dir = progen_model_dir.rstrip("/") + f"/progen2-{PROGEN2_VARIANT[model_key]}"
        model = ProGenForCausalLM.from_pretrained(
            model_dir, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device)
        model.eval()

        with open(progen_model_dir.rstrip("/") + "/tokenizer.json", "r") as f:
            tokenizer = Tokenizer.from_str(f.read())
        tokenizer.no_padding()

        df_o = get_iterative_masked_progen2(model, tokenizer, sequence, sequence_name, device_str=device_str, verbose=verbose)
        res_df = augment_masked_df_with_site_entropy(df_o)
        return res_df

    if model_key in PROTT5_KEY_TO_MODEL_STRING:
        model_name = PROTT5_KEY_TO_MODEL_STRING[model_key]
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, cache_dir=huggingface_cache_dir)
        model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=huggingface_cache_dir).eval().to(device)

        df_o = get_iterative_masked_prott5(model, tokenizer, sequence, sequence_name, device_str=device_str, verbose=verbose)
        res_df = augment_masked_df_with_site_entropy(df_o)
        return res_df

    using_direct_loading = True
    if model_key in key_to_model_string.keys():
        using_direct_loading = False

    if force_direct_loading:
        using_direct_loading = True



    use_esm = False
    if "esm" in model_key:
        use_esm = True

    if using_direct_loading:
        if direct_loading_model_path is None:
            raise ValueError(
                f"model_key '{model_key}' did not match esm/protbert/carp/progen2/prott5, "
                "and no direct_loading_model_path was given for direct .pt loading."
            )
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
                if model_key == "esm_2_15B":
                    # 15B is too large to auto-download on demand; requires the
                    # weights already cached locally (huggingface_cache_dir).
                    tokenizer = EsmTokenizer.from_pretrained(model_name, do_lower_case=False, cache_dir=huggingface_cache_dir, local_files_only=True)
                    model = EsmForMaskedLM.from_pretrained(model_name, cache_dir=huggingface_cache_dir, local_files_only=True)
                else:
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
