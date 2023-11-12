import pickle as pkl
import pandas as pd


if __name__ == '__main__':
    glf_cr_list = pd.read_csv("../splits/outdated_train_val_test_patches.csv", header=None)
    def insert_s2(x):
        parts = x.split("_")
        parts.insert(2, "s2")
        return "_".join(parts)
    glf_cr_list[4] = glf_cr_list[4].apply(insert_s2)
    glf_cr_train = glf_cr_list[glf_cr_list[0] == 1]
    glf_cr_val = glf_cr_list[glf_cr_list[0] == 2]
    glf_cr_test = glf_cr_list[glf_cr_list[0] == 3]

    train_list_new = glf_cr_train[4].tolist()
    val_list_new = glf_cr_val[4].tolist()
    test_list_new = glf_cr_test[4].tolist()

    # Save new lists as pickle and txt
    pkl.dump(train_list_new, open("../splits/train_list.pkl", "wb"))
    pkl.dump(val_list_new, open("../splits/val_list.pkl", "wb"))
    pkl.dump(test_list_new, open("../splits/test_list.pkl", "wb"))

    with open("../splits/train_list.txt", "w") as f:
        f.write("\n".join(train_list_new))
    with open("../splits/val_list.txt", "w") as f:
        f.write("\n".join(val_list_new))
    with open("../splits/test_list.txt", "w") as f:
        f.write("\n".join(test_list_new))

    #glf_cr_list = pd.read_csv("../splits/outdated_train_val_test_patches.csv")
    #glf_cr_list = glf_cr_list[glf_cr_list[glf_cr_list.columns[0]] == 3]
    #glf_cr_list = glf_cr_list[glf_cr_list.columns[4]].tolist()
    #glf_cr_list.sort()

    #sample_list = pkl.load(open("../splits/test_list.pkl", "rb"))
    #sample_list = [s.replace("s2_", "") for s in sample_list]
    #sample_list.sort()

    #sample_list_scenes = list(set(["_".join(s.split("_")[0:-1]) for s in sample_list]))

    #intersection = set(sample_list).intersection(set(glf_cr_list))
    #diff1 = set(sample_list).difference(set(glf_cr_list))
    #diff2 = set(glf_cr_list).difference(set(sample_list))
