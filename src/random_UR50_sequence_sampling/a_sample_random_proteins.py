import random 
random.seed(0)
import gzip 

def read_fasta(fn, return_headers=False):
    seq_list = []
    header_list = []

    with gzip.open(fn, "rt") as f:
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


if __name__ == "__main__" : 
    fasta_path = "uniref50_042021.fasta.gz"
    header_list, seq_list = read_fasta(fasta_path,return_headers = True)

    num_to_sample = 50000
    total_positions = list(range(0, len(header_list)))
    # sample without replacement 
    positions_to_sample = random.sample(total_positions, num_to_sample)

    num_needed = 25000
    # only include 25000 of those less than 1022 in length 
    valid = 0 
    with open("randomly_chosen_proteins_UR50.fasta", "w") as f2:
        for p in positions_to_sample:
            header = header_list[p]
            seq = seq_list[p]
            if len(seq) < 1022:

                f2.write(">" + header + "\n")
                f2.write(seq + "\n")
                valid += 1
            
            if valid == num_needed:
                break
    f2.close()



