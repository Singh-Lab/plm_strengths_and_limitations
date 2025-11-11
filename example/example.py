from plm_masking import get_iterative_masked_df

sequence_name = "P155_HUMAN"
sequence = "MEMALMVAQTRKGKSVV"


model_key = "esm_2_8M"
seq_results = get_iterative_masked_df(model_key, sequence, sequence_name, huggingface_cache_dir=huggingface_cache_dir, verbose = True)
