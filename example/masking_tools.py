import re
import warnings
from argparse import Namespace
from pathlib import Path
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
import torch
import esm
from esm.model.esm2 import ESM2
import copy

torch.serialization.add_safe_globals([Namespace])

# The direct-.pt-checkpoint loading code below (load_model_and_alphabet_core,
# _load_model_and_alphabet_core_v1/v2, load_model_and_alphabet_local, and
# has_emb_layer_norm_before) is adapted from facebookresearch/esm's esm/pretrained.py

def _has_regression_weights(model_name):
    """Return whether we expect / require regression weights;
    Right now that is all models except ESM-1v, ESM-IF, and partially trained ESM2 models"""
    return False # we aren't doing contact predictions 
    
    # uncomment the below line to go back to original v
    # return not ("esm1v" in model_name or "esm_if" in model_name or "270K" in model_name or "500K" in model_name)


def load_model_and_alphabet(model_name):
    if model_name.endswith(".pt"):  # treat as filepath
        return load_model_and_alphabet_local(model_name)
    else:
        return load_model_and_alphabet_hub(model_name)
    


def has_emb_layer_norm_before(model_state):
    """Determine whether layer norm needs to be applied before the encoder"""
    return any(k.startswith("emb_layer_norm_before") for k, param in model_state.items())


def _load_model_and_alphabet_core_v1(model_data):
    import esm  # since esm.inverse_folding is imported below

    alphabet = esm.Alphabet.from_architecture(model_data["args"].arch)

    if model_data["args"].arch == "roberta_large":
        # upgrade state dict
        pra = lambda s: "".join(s.split("encoder_")[1:] if "encoder" in s else s)
        prs1 = lambda s: "".join(s.split("encoder.")[1:] if "encoder" in s else s)
        prs2 = lambda s: "".join(
            s.split("sentence_encoder.")[1:] if "sentence_encoder" in s else s
        )
        model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
        model_state = {prs1(prs2(arg[0])): arg[1] for arg in model_data["model"].items()}
        model_state["embed_tokens.weight"][alphabet.mask_idx].zero_()  # For token drop
        model_args["emb_layer_norm_before"] = has_emb_layer_norm_before(model_state)
        model_type = esm.ProteinBertModel

    elif model_data["args"].arch == "protein_bert_base":

        # upgrade state dict
        pra = lambda s: "".join(s.split("decoder_")[1:] if "decoder" in s else s)
        prs = lambda s: "".join(s.split("decoder.")[1:] if "decoder" in s else s)
        model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
        model_state = {prs(arg[0]): arg[1] for arg in model_data["model"].items()}
        model_type = esm.ProteinBertModel
    elif model_data["args"].arch == "msa_transformer":

        # upgrade state dict
        pra = lambda s: "".join(s.split("encoder_")[1:] if "encoder" in s else s)
        prs1 = lambda s: "".join(s.split("encoder.")[1:] if "encoder" in s else s)
        prs2 = lambda s: "".join(
            s.split("sentence_encoder.")[1:] if "sentence_encoder" in s else s
        )
        prs3 = lambda s: s.replace("row", "column") if "row" in s else s.replace("column", "row")
        model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
        model_state = {prs1(prs2(prs3(arg[0]))): arg[1] for arg in model_data["model"].items()}
        if model_args.get("embed_positions_msa", False):
            emb_dim = model_state["msa_position_embedding"].size(-1)
            model_args["embed_positions_msa_dim"] = emb_dim  # initial release, bug: emb_dim==1

        model_type = esm.MSATransformer

    elif "invariant_gvp" in model_data["args"].arch:
        import esm.inverse_folding

        model_type = esm.inverse_folding.gvp_transformer.GVPTransformerModel
        model_args = vars(model_data["args"])  # convert Namespace -> dict

        def update_name(s):
            # Map the module names in checkpoints trained with internal code to
            # the updated module names in open source code
            s = s.replace("W_v", "embed_graph.embed_node")
            s = s.replace("W_e", "embed_graph.embed_edge")
            s = s.replace("embed_scores.0", "embed_confidence")
            s = s.replace("embed_score.", "embed_graph.embed_confidence.")
            s = s.replace("seq_logits_projection.", "")
            s = s.replace("embed_ingraham_features", "embed_dihedrals")
            s = s.replace("embed_gvp_in_local_frame.0", "embed_gvp_output")
            s = s.replace("embed_features_in_local_frame.0", "embed_gvp_input_features")
            return s

        model_state = {
            update_name(sname): svalue
            for sname, svalue in model_data["model"].items()
            if "version" not in sname
        }

    else:
        raise ValueError("Unknown architecture selected")

    model = model_type(
        Namespace(**model_args),
        alphabet,
    )

    return model, alphabet, model_state


def _load_model_and_alphabet_core_v2(model_data):
    def upgrade_state_dict(state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    state_dict = upgrade_state_dict(state_dict)
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    model = ESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        alphabet=alphabet,
        token_dropout=cfg.token_dropout,
    )
    return model, alphabet, state_dict


def entropy(lst):
    e = list(map(lambda x: x * np.log2(x), lst))
    e = -1 * np.sum(e)
    return e

def augment_masked_df_with_site_entropy(df, limit_normal_aas = True, return_entropy = True):
    regular_aas = ['P','S','L','A','V','G','T','E','R','Q','D','F','I','H','Y','N','K','C','W','M']
    df["name"] = df["ref_aa"] + df["aa_pos"].astype(str) + df["token_str"]
    df = df.loc[df["token_str"].isin(regular_aas)]
    variant_to_s1 = {}
    variant_to_adj_prob = {}
    variant_to_site_entropy = {}
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
            adj_prob_list.append(row["adj_score"] / valid_prob_total)
        ent_val = entropy(adj_prob_list)

        for name in sub_df["name"].values: variant_to_site_entropy[name] = ent_val

    if return_entropy:
        df["site_entropy"] = df["name"].apply(lambda x : variant_to_site_entropy[x])

    df["s1"] = df["name"].apply(lambda x : variant_to_s1[x])
    df["adj_score"] = df["name"].apply(lambda x : variant_to_adj_prob[x])

    df = df.set_index("name")
    return df

def load_model_and_alphabet_core(model_name, model_data, regression_data=None):
    if regression_data is not None:
        model_data["model"].update(regression_data["model"])

    if model_name.startswith("esm2"):
        model, alphabet, model_state = _load_model_and_alphabet_core_v2(model_data)
    else:
        model, alphabet, model_state = _load_model_and_alphabet_core_v1(model_data)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    if regression_data is None:
        expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
        error_msgs = []
        missing = (expected_keys - found_keys) - expected_missing
        if missing:
            error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
        unexpected = found_keys - expected_keys
        if unexpected:
            error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

        if error_msgs:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        if expected_missing - found_keys:
            warnings.warn(
                "Regression weights not found, predicting contacts will not produce correct results."
            )

    model.load_state_dict(model_state, strict=regression_data is not None)

    return model, alphabet

    
def load_model_and_alphabet_local(model_location):
    """Load from local path. The regression weights need to be co-located"""
    model_location = Path(model_location)
    model_data = torch.load(str(model_location), map_location="cpu")
    model_name = model_location.stem
    if _has_regression_weights(model_name):
        regression_location = str(model_location.with_suffix("")) + "-contact-regression.pt"
        regression_data = torch.load(regression_location, map_location="cpu")
    else:
        regression_data = None
    return load_model_and_alphabet_core(model_name, model_data, regression_data)


def get_iterative_masked_seqeunce(model, alphabet, batch_converter, sequence, seq_name, device_str="cuda"):
    input_df = pd.DataFrame([
            ("seq1",seq_name, sequence, len(sequence))
        ], 
        columns = ['id','gene','seq','length']
    )
    for gname in input_df["id"].values:
        dt = [(gname + '_WT',input_df[input_df.id==gname].seq.values[0])]

    # BOS is always prepended; EOS is only appended for some architectures
    # (e.g. ESM-1b/roberta_large and ESM-2 via the "ESM-1b" alphabet) -- protein_bert_base
    # ESM-1 checkpoints have no EOS. Slice both ends off using the alphabet's own flag
    # rather than assuming "no EOS" universally.
    end_idx = -1 if alphabet.append_eos else None

    res_vals = []
    for i in range(0, len(sequence)):
        ref_aa = sequence[i]

        batch_labels, batch_strs, batch_tokens = batch_converter(copy.deepcopy(dt))

        bt_2 = copy.deepcopy(batch_tokens)

        # one indexed to account for begin token
        bt_2[0, i+1] = alphabet.mask_idx

        # was cuda in place od device _str
        results = torch.softmax(model(bt_2.to(device_str), repr_layers=[34], return_contacts=False)["logits"], dim =-1)
        results = pd.DataFrame(results[0,:,:].cpu().detach().numpy()[1:end_idx,:], columns=alphabet.all_toks, index=list(input_df[input_df.id==gname].seq.values[0])).T
        results.columns = [j.split('.')[0]+' '+str(i+1) for i,j in enumerate(results.columns)]

        col_of_interest = ref_aa + " " + str(i + 1)

        masked_distribution = results[col_of_interest]


        for index, item in masked_distribution.items():
            name = ref_aa + str(i+1) + index 
            res_vals.append({
                "aa_pos" : i + 1,
                "score" : item, 
                "token_str" : index, 
                "ref_aa" : ref_aa, 
                "gene" : seq_name,
                "name" : name
            })

    df_out = pd.DataFrame(res_vals)
    return df_out


def get_iterative_masked_carp(model, collater, sequence, seq_name, device_str="cuda"):
    """Ported from iteratively_mask_sequences.py (iteratively_mask_sequence),
    parameterized instead of relying on module globals."""
    CAN_AAS = 'ACDEFGHIKLMNPQRSTVWY'
    AMB_AAS = 'BZX'
    OTHER_AAS = 'JOU'
    ALL_AAS = CAN_AAS + AMB_AAS + OTHER_AAS
    STOP = '*'
    GAP = '-'
    MASK = '#'
    START = '@'
    SPECIALS = STOP + GAP + MASK + START
    PROTEIN_ALPHABET = ALL_AAS + SPECIALS

    index_to_char = {i: c for i, c in enumerate(PROTEIN_ALPHABET)}

    results = []
    for i in range(len(sequence)):
        aa_pos = i + 1

        seq_2 = list(sequence)
        ref_aa = seq_2[i]
        seq_2[i] = MASK
        seq_2 = "".join(seq_2)

        x = collater([[seq_2]])[0].to(device_str)
        with torch.no_grad():
            output = model(x, logits=True)
        probs = torch.softmax(output["logits"], dim=-1)
        wanted_col = probs[0, i, :].cpu().detach().numpy()

        for alph_idx, prob in enumerate(wanted_col):
            results.append({
                "score": prob,
                "token": alph_idx,
                "token_str": index_to_char[alph_idx],
                "aa_pos": aa_pos,
                "ref_aa": ref_aa,
                "gene": seq_name,
            })

    return pd.DataFrame(results)


def get_iterative_masked_progen2(model, tokenizer, sequence, seq_name, device_str="cuda", verbose=False):
    """Ported from progen2_iteratively_mask_sequences.py.

    NOTE: ProGen2 is a causal (autoregressive) LM, not a masked LM. Each position's
    score distribution is computed from only the *preceding* sequence context (the
    model has never seen anything after that position) -- this is not the same thing
    as ESM/CARP/ProtT5's bidirectional mask-infilling, even though the output shape
    and downstream site-entropy math are the same.
    """
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    # 2 tokens short of the logits shape for the medium/large/bfd checkpoints
    id_to_token[30] = "<MISSING_1>"
    id_to_token[31] = "<MISSING_2>"

    results = []
    for idx in range(len(sequence)):
        aa_pos = idx + 1
        wt_aa = sequence[idx]

        prompt = "1" + sequence[:idx]
        input_ids = torch.tensor(tokenizer.encode(prompt).ids).to(device_str)
        with torch.no_grad():
            model_output = model(input_ids, output_hidden_states=True)
        next_token_logits = model_output.logits[-1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1).detach().cpu().numpy()

        if verbose:
            print(seq_name, aa_pos, "/", len(sequence))

        for i, token_prob in enumerate(next_token_probs):
            if i > 31:
                continue
            results.append({
                "score": token_prob,
                "token": i,
                "token_str": id_to_token[i],
                "aa_pos": aa_pos,
                "ref_aa": wt_aa,
                "gene": seq_name,
            })

    return pd.DataFrame(results)


def get_iterative_masked_prott5(model, tokenizer, sequence, seq_name, device_str="cuda", verbose=False):
    """Ported from iteratively_mask_sequence.py.

    Uses T5's span-corruption objective (<extra_id_0>) rather than a single [MASK]
    token: the masked position is replaced with the sentinel, and the decoder is fed
    the correct prefix of preceding residues to predict the next (masked) token.
    """
    decoder_start_id = model.config.decoder_start_token_id
    extra0 = "<extra_id_0>"

    amino_acids = list("ACDEFGHIKLMNPQRSTVWYX")
    aa_to_id = {}
    for aa in amino_acids:
        ids = tokenizer.encode(" " + aa, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"AA {aa} not 1 token")
        aa_to_id[aa] = ids[0]

    # matches the original script: non-canonical AAs are folded into X
    clean_seq = sequence.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")

    results = []
    with torch.no_grad():
        for idx in range(len(clean_seq)):
            aa_pos = idx + 1
            wt_aa = clean_seq[idx]

            if verbose:
                print(seq_name, aa_pos, "/", len(clean_seq))

            toks = list(clean_seq)
            toks[idx] = extra0
            enc_text = " ".join(toks)
            enc = tokenizer(enc_text, return_tensors="pt", add_special_tokens=True)
            enc = {k: v.to(device_str) for k, v in enc.items()}

            prefix_ids = [aa_to_id[aa] for aa in clean_seq[:idx]]
            dec_in = torch.tensor([[decoder_start_id] + prefix_ids], device=device_str)

            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask", None),
                decoder_input_ids=dec_in,
            )
            mask_logits = out["logits"][0, -1]
            probs = torch.softmax(mask_logits, dim=-1).detach().cpu().numpy()

            for i, token_prob in enumerate(probs):
                token_str = tokenizer.convert_ids_to_tokens(i)
                if len(token_str) == 2:  # e.g. "_M" -> "M"
                    token_str = token_str[1:]
                results.append({
                    "score": token_prob,
                    "token": i,
                    "token_str": token_str,
                    "aa_pos": aa_pos,
                    "ref_aa": wt_aa,
                    "gene": seq_name,
                })

    return pd.DataFrame(results)
