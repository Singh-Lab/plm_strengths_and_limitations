BLAST+ version 2.14.0 was used for homolog searches 

Information on this release can be found [here](https://www.ncbi.nlm.nih.gov/books/NBK131777/#Blast_ReleaseNotes.BLAST_2_14_0_April_25).

This version of BLAST+ executables can be downloaded [here](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/) along with other versions. 


Source files needed to build databases : 



Running homolog searches requires building the database from a fasta file 

```console
makeblastdb -in /path/to/fasta/uniref50_042021.fasta -dbtype prot -title uniref50_042021 -out uniref50_042021
```
