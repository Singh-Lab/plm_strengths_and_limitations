import glob 
import pandas as pd 
import sys 

if __name__ == "__main__":
    paths = [
        "../embeddings/MODEL/",
        "../randseq_embeddings/MODEL/",
        "../markov_based_embeddings/MODEL/window_size_1/",
        "../markov_based_embeddings/MODEL/window_size_2/",
        "../markov_based_embeddings/MODEL/window_size_3/",
        "../markov_based_embeddings/MODEL/window_size_4/",
        "../markov_based_embeddings/MODEL/window_size_5/",
        "../markov_based_embeddings/MODEL/window_size_6/",
        "../markov_based_embeddings/MODEL/window_size_7/",
        "../markov_based_embeddings/MODEL/window_size_8/",
        "../markov_based_embeddings/MODEL/window_size_9/",
        "../markov_based_embeddings/MODEL/window_size_10/",
        "../markov_based_embeddings/MODEL/window_size_11/",
        "../markov_based_embeddings/MODEL/window_size_12/"
    ]

    names = [
        "Orphan25",
        "randseq",
        "markov_1",
        "markov_2",
        "markov_3",
        "markov_4",
        "markov_5",
        "markov_6",
        "markov_7",
        "markov_8",
        "markov_9",
        "markov_10",
        "markov_11",
        "markov_12"
    ]

    test_model = "esm_2"

    resutls = []
    for name, path in zip(names, paths):
        path_to_glob = path.replace("MODEL", test_model)
        files = glob.glob(path_to_glob + "*.pkl")
        for fn in files:
            path_to_report = fn.replace(test_model, "MODEL")
            prot_name = path_to_report.replace(path, "").replace(".pkl", "")
            if name == "randseq":
                prot_name = "randseq_" + prot_name
            elif name == "Orphan25":
                prot_name = "Orphan25_" + prot_name

            resutls.append([name, prot_name, path_to_report])

    res = pd.DataFrame(resutls, columns = ["class_name", "protein_name", "path"])
    res.to_csv("distance_comps_reference_file.csv")
