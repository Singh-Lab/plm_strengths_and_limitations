import copy

import numpy as np
import torch


def extract_embedding_huggingface(seq, model, tokenizer, device_str="cuda"):
    """Ported from extract_embeddings_transformer.py (extract_embedding_for_sequence).
    Used for ESM-1b/1v/2, ProtBERT."""
    wt_seq = copy.deepcopy([*seq])
    tokenized_input = tokenizer(wt_seq, return_tensors="pt", is_split_into_words=True).to(device_str)
    with torch.no_grad():
        output = model(**tokenized_input, output_attentions=False)
    final_hidden = output.last_hidden_state.cpu().detach().numpy()[0]

    mean_pooled = np.mean(final_hidden[1:-1, :], axis=0)
    cls_pooled = final_hidden[0]
    return {"mean_pooled": mean_pooled, "cls_pooled": cls_pooled}


def extract_embedding_prott5(seq, model, tokenizer, device_str="cuda"):
    """Ported from extract_embeddings_transformer.py
    (extract_prott5_xl_embedding_for_sequence). T5 encoder-only, no CLS token."""
    spaced = [' '.join(list(seq))]
    tokenized_input = tokenizer(spaced, return_tensors="pt", add_special_tokens=True)['input_ids'].to(device_str)
    with torch.no_grad():
        output = model(tokenized_input)
    final_hidden = output.last_hidden_state.cpu().detach().numpy()[0]
    final_hidden = final_hidden[:-1, ]  # drop end-of-sequence token

    mean_pooled = np.mean(final_hidden, axis=0)
    return {"mean_pooled": mean_pooled}


def extract_embedding_direct_loading(model, alphabet, batch_converter, sequence, seq_name, device_str="cuda"):
    """Ported from extract_embeddings_direct_loading.py.

    That script picks repr_layers from a hardcoded per-model-key dict; here we use
    len(model.layers) (the actual transformer stack depth) instead, so this works for
    any ESM-1 checkpoint without needing that lookup table."""
    dt = [(seq_name, sequence)]
    _, _, batch_tokens = batch_converter(dt)

    layer_num = len(model.layers)
    with torch.no_grad():
        model_res = model(batch_tokens.to(device_str), repr_layers=[layer_num], return_contacts=False)
    final_hidden = model_res["representations"][layer_num]

    # BOS is always prepended; EOS is only appended for some architectures (e.g.
    # ESM-1b/roberta_large and ESM-2 via the "ESM-1b" alphabet). Slice both ends off
    # using the alphabet's own flag rather than assuming "no EOS" universally.
    end_idx = -1 if alphabet.append_eos else None
    aa_reps = final_hidden[0, 1:end_idx, :].cpu().numpy()
    cls_pooled = final_hidden[0, 0, :].cpu().numpy()
    mean_pooled = np.mean(aa_reps, axis=0)
    return {"mean_pooled": mean_pooled, "cls_pooled": cls_pooled}


def extract_embedding_progen2(model, tokenizer, sequence, device_str="cuda"):
    """Ported from extract_embeddings_progen2.py."""
    prompt = "1" + sequence
    input_ids = torch.tensor(tokenizer.encode(prompt).ids).to(device_str)
    with torch.no_grad():
        model_output = model(input_ids, output_hidden_states=True)
    final_hidden = model_output.hidden_states[-1].detach().cpu().numpy()

    cls_pooled = final_hidden[0, :]
    mean_pooled = np.mean(final_hidden[1:, :], axis=0)
    return {"mean_pooled": mean_pooled, "cls_pooled": cls_pooled}


def extract_embedding_carp(sequence, model, collater, layer_num, device_str="cuda"):
    """Ported from extract_embeddings_carp.py (get_final_hidden_state).
    No CLS token for CARP -- mean-pooled only."""
    x = collater([[sequence]])[0]
    with torch.no_grad():
        rep = model(x.to(device_str), repr_layers=[layer_num], logits=True)
    final_hidden = rep["representations"][layer_num][0].detach().cpu().numpy()

    mean_pooled = np.mean(final_hidden, axis=0)
    return {"mean_pooled": mean_pooled}
