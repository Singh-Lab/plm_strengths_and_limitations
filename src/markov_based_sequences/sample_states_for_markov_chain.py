import gzip 
import sys
import copy
import json
import time

from config import llm_review_path, home_path

def get_state_probabilities(sequence_file, window_size):
    counter = 0

    initial_probabilities = {}
    window_primer_to_next_aa_obs = {}


    regular_aas = {'P','S','L','A','V','G','T','E','R','Q','D','F','I','H','Y','N','K','C','W','M'}
    regular_aa_initial = {aa : 0 for aa in regular_aas}

    ## making all combinations of window_size aas
    
    old_pairs = copy.deepcopy(regular_aas)

    # starting from size of 1 (old_pairs is 20 long), need to do window_size - 1 iterations 
    for i in range(window_size - 1):
        kmers = set()
        for j in old_pairs:
            for aa in regular_aas:
                kmers.add(j + aa)
        old_pairs = kmers
 
    all_kmers_initial = {kmer : 0 for kmer in old_pairs}
    all_kmers_transitions = {kmer : copy.deepcopy(regular_aa_initial) for kmer in old_pairs}

    t1 = time.time()
    with gzip.open(sequence_file,'rt') as f: 
        header = True 
        while header:
            counter += 1
            if counter % 1000000 == 0:
                print("counter ", counter , time.time() - t1)
            header = f.readline().replace("\n", "")
            sequence = f.readline().replace("\n", "")

            if header is None or header == "":
                break 
            
            if sequence is None or sequence == "":
                break

            irregular_aa = False
            for aa in sequence:
                if aa not in regular_aas:
                    irregular_aa = True
            if irregular_aa:
                continue  
           
            init_window = sequence[: window_size]           
            all_kmers_initial[init_window] += 1
          
            current_window = init_window
            for aa in [*sequence[window_size :]]:
                all_kmers_transitions[current_window][aa] += 1
                current_window = current_window[1:] + aa


    f.close()
    t2 = time.time()

    print(counter, t2 - t1)

    return all_kmers_initial, all_kmers_transitions




if __name__ == "__main__":
    window_size = int(sys.argv[1])
    sequence_file = llm_review_path + "model_source_datasets/uniref50_042021/uniref50_042021.fasta.gz"
    all_kmers_initial, all_kmers_transitions = get_state_probabilities(sequence_file, window_size)
    res_dict = {"initial" : all_kmers_initial, "transitions" : all_kmers_transitions}
    
    with open("./markov_aa_results/observations_window_" + str(window_size) + ".json", "w") as f1:
        json.dump(res_dict, f1)
    f1.close()

