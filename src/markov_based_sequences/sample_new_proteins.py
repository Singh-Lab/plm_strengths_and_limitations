import numpy as np 
import json 
import copy 
import sys
import random 
np.random.seed(0)

def convert_count_dict_to_prob_lists(d):
    total_obs = np.sum(list(d.values()))
    s_list = []
    p_list = []
    for k,v in d.items():
        s_list.append(k)
        p_list.append(v / total_obs)
    return s_list, p_list

def get_seqeunce(needed_length, initial_states, transitions):
    s_list_init, p_list_init = convert_count_dict_to_prob_lists(initial_states)
    random_initial = np.random.choice(s_list_init, size = 1, p = p_list_init)[0]
    total_string = copy.deepcopy(random_initial)
    current_state = random_initial
    current_len = len(current_state)

    while current_len < needed_length:
        transition_dict = transitions[current_state]
        s_list, p_list = convert_count_dict_to_prob_lists(transition_dict)
        sampled_aa = np.random.choice(s_list, size = 1, p = p_list)[0]
        total_string = total_string + sampled_aa
        current_state = current_state[1:] + sampled_aa
        current_len += 1

    return total_string

if __name__ == "__main__":
    window_size = int(sys.argv[1])
    res_file = f"./markov_aa_results/observations_window_{window_size}.json"

    with open(res_file, "r") as f: 
        markov_results  = json.load(f)
    f.close()

    initial_states = markov_results["initial"]
    transitions = markov_results["transitions"]

    with open("randseq_generations.fasta", "r") as f1:
        lines = f1.readlines()
    f1.close()

    lines = list(map(lambda x : x.replace("\n", ""), lines))

    entries = np.reshape(lines, (25,2))
    randseq_sequences = entries[:, 1]
    needed_lengths = list(map(lambda x : len(x), randseq_sequences))

    with open("./markov_based_seqs/markov_sequences_UR50_based_window_size_" + str(window_size) + ".fasta", "w") as f2:
        for i, needed_seq_len in enumerate(needed_lengths): 
            seq1 = get_seqeunce(needed_seq_len, initial_states, transitions)
            seq_name = f">window_size_{window_size}_sequence_{i};UR50 based markov_seq;length " + str(needed_seq_len) 
            f2.write(seq_name + "\n") 
            f2.write(seq1 + "\n")
    f2.close()


