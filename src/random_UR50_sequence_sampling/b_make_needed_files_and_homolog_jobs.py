import pandas as pd
import glob
import os
import numpy as np 
from config import home_path, llm_review_path

def read_fasta(fn, return_headers=False):
    seq_list = []
    header_list = []

    with open(fn, "r") as f:
        line = True
        header = None
        running_seq = ""
        while line:
            line = f.readline().replace("\n", "")
            if line is None or len(line) == 0:
                header_list.append(header.replace(">", ""))
                seq_list.append(running_seq)
                running_seq = ""
                continue

            if line[0] == ">":
                if header is not None:
                    seq_list.append(running_seq)
                    header_list.append(header.replace(">", ""))
                    running_seq = ""
                    header = line
                else:

                    header = line

            else:
                running_seq = running_seq + line

    if return_headers:
        return header_list, seq_list
    return seq_list

if __name__ == "__main__":
    fasta_path = "randomly_chosen_proteins_UR50.fasta"
    header_list, seq_list = read_fasta(fasta_path, return_headers = True)

    name_set = set()
    for header, sequence in zip(header_list, seq_list):
        name = header.replace(">", "").split(" ")[0]
        out_fn = "./fastas/" + name + ".fasta"
        name_set.add(name)
        with open(out_fn, "w") as f0:
            f0.write(">" + name + "\n")
            f0.write(sequence)
        f0.close()

    assert len(name_set) == len(header_list)

    blast_path = home_path + "blast/ncbi-blast-2.14.0+/bin/blastp"
    db_path = llm_review_path + "model_source_datasets/uniref50_042021/uniref50_042021"

    cwd = os.getcwd()
    out_dir = cwd + "/alignment_results/"
    print(cwd)
    fasta_files = glob.glob(cwd + "/fastas/" + "*.fasta")
    
    num_splits = 250 # 25000 total, 100 per file
    file_lists = np.array_split(np.array(fasta_files), num_splits)
   

    for i, lst in enumerate(file_lists):
        job_fn = cwd + "/jobs/run_alignments_random_sample_" + str(i) + ".sh"

        with open(job_fn, "w") as f0:
            header = "#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --mem-per-cpu=128GB\n#SBATCH --time=12:00:00\n"
            header_2 = "#SBATCH --out=" + cwd + "/job_outputs/alignment_" + str(i) + ".out\n"
            f0.write(header)
            f0.write(header_2)
            f0.write("\n\n")

            for fn in lst:
                identifier = fn.replace(cwd + "/fastas/", "").replace(".fasta", "")
                out_fn = out_dir + identifier + ".json"
                command = [
                    blast_path,
                    "-db=" + db_path,
                    "-query="+fn,
                    "-max_target_seqs=1000000000",
                    "-evalue=200",
                    "-outfmt=15",
                    "-task=blastp-fast",
                    "-out="+out_fn,
                ]
                command = " ".join(command)
                f0.write(command + "\n\n\n")
        f0.close()

        os.system("sbatch " + job_fn)

