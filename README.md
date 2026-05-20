## Repository for Systematic evaluation of protein language models highlights strengths and limitations
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/DOI/TBD/zenodo.TBD.svg)](https://doi.org/)




### Downloads
MAPLE-based selections for missense variant impacts throughout the human proteome available at: 
https://drive.google.com/file/d/11gXT_6FRHxsYpvBD5UdMtPAxytrF2Yj0/view?usp=sharing

---


### Example usage for generating masked languge model results with positional entropy 
Example code located within ``./examples/example.py`` 



By default the function will attempt to load the model from Huggingface. See ``./examples/plm_masking.py`` for Huggingface repositories and models supported.

```python 
from plm_masking import get_iterative_masked_df

sequence_name = "P155_HUMAN"
sequence = "MEMALMVAQTRKGKSVV"

model_key = "esm_2_8M"
seq_results = get_iterative_masked_df(model_key, sequence, sequence_name)

```

Alternatively, downloaded torch (.pt) models can be loaded using the direct loading option with the path to the downloaded model

```python 
from plm_masking import get_iterative_masked_df

sequence_name = "P155_HUMAN"
sequence = "MEMALMVAQTRKGKSVV"


pt_model_path = "./esm1_t6_43M_UR50S.pt"
model_key = "esm1_t6_43M_UR50S"
seq_results = get_iterative_masked_df(model_key, sequence, sequence_name, direct_loading_model_path = pt_model_path)

```

resulting ``seq_results`` dataframe has the following format :

```
         score  token token_str  aa_pos ref_aa        gene  site_entropy        s1  adj_score
name                                                                                         
M1M   0.657993     20         M       1      M  P155_HUMAN      2.298782  0.000000   0.664377
M1L   0.035542      4         L       1      M  P155_HUMAN      2.298782 -2.918480   0.035887
M1V   0.032935      7         V       1      M  P155_HUMAN      2.298782 -2.994645   0.033255
M1E   0.030001      9         E       1      M  P155_HUMAN      2.298782 -3.087956   0.030292
M1A   0.027771      5         A       1      M  P155_HUMAN      2.298782 -3.165202   0.028040

```

`name` is the variant name in the form <ref_aa><sequence_position><alt_aa><br />
`token` is the index of the model's alphabet corresponding to the listed token (amino acid)<br />
`token_str` is the amino acid corresponding to the index, the alternate amino acid in the name<br />
`aa_pos` is the position of the masked amino acid<br />
`ref_aa` is the reference amino acid in the masked position<br />
`score` is the probability of the token filling the maked position<br />
`gene` is the provided name of the protein filled with the `sequence_name` argument <br />
`adj_score` is the probability of each of the tokens filling the masked position when comparing the raw `score` to only the 20 canonical amino acid probabilites <br />
`s1` is the variant effect prediction score <br />
`site_entropy` is the Shannon Entropy of the position. This will be the same value for all rows with the same `aa_pos`<br /> <br />


Running `example.py` should take only a few secconds on a machine where cuda is available. 


--- 

Package requirements to run ``example.py`` 

```
pandas==1.5.2
numpy==1.23.4
trasnformers>=4.25.1
torch>=1.12.1
sequence-models==1.8.0
```


