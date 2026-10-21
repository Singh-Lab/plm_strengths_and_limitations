BLAST+ version 2.14.0 was used for homolog searches 

Information on this release can be found [here](https://www.ncbi.nlm.nih.gov/books/NBK131777/#Blast_ReleaseNotes.BLAST_2_14_0_April_25).

This version of BLAST+ executables can be downloaded [here](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/) along with other versions. 


Source files needed to build databases for UR50/90/100 are uploaded in the zenodo repository. 



Running homolog searches requires building the database from a fasta file 

```console
makeblastdb -in /path/to/fasta/uniref50_042021.fasta -dbtype prot -title uniref50_042021 -out uniref50_042021
```


When running the search we had used an initial E-value of 200, a larger value than typically used. This was to gauge the effect of different E-values affecting the search results from databases of various sizes when applying different thresholds afterwards. We also used a `max_target_seqs` argument that was larger than the size of the database for this reason.  We ultimately found that an E-value of 10^{-5} was of minimal effect when filtering the hits based on our other criteria which was considerably strict, but included this threshold in our filtering.


```console

blastp -db uniref50_042021 -query sample_sequence.fasta -evalue=200 -max_target_seqs=YOUR_MAX_TARGET_SEQS -outfmt=15 -task=blastp-fast -out=sample_sequence.json
```

Homologs were filtered based on `GAP_FRAC` and `ALN_FRAC` of 0.75. `GAP_FRAC` was based on reported `"gaps"` divided by the `"align_len"` from BLAST, and `ALN_FRAC` based on the reported `align_len` divided by the `query_len` from BLAST.  

```python

with open("sample_sequence.json", "r") as f2:
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

GAP_FRAC = 0.75
ALN_FRAC = 0.75
EVAL = 1e-5

hsp_df_filtered = hsp_df.loc[
    (hsp_df["percent_gaps"] <= GAP_FRAC) &
    (hsp_df["alignment_frac"] >= ALN_FRAC) &
    (hsp_df["evalue"] <= EVAL)
]
```
