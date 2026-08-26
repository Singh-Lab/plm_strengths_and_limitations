from plm_embeddings import get_embedding

id_ = "P155_HUMAN"
sequence = "MEMALMVAQTRKGKSVV"

model_key = "esm_2_8M"
embedding = get_embedding(model_key, sequence, id_)

print("mean_pooled shape:", len(embedding["mean_pooled"]))
print(embedding["mean_pooled"][:5])
