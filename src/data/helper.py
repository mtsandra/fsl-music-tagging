import numpy as np
import pandas as pd

ANNOTATION_PATH = "/ssddata/tma98/data/mtat/annotations_final.csv"
CLIP_INFO_PATH = "/ssddata/tma98/data/mtat/clip_info_final.csv"
T50_TRAIN = "./data/train.npy"
T50_VALID = "./data/valid.npy"
T50_TEST = "./data/test.npy"

# sort the tags by frequency

df = pd.read_csv(ANNOTATION_PATH, sep='\t')
# drop songs with no tags
mask = (df.iloc[:,1:-1]==0).all(axis=1)
df = df.loc[~mask]

tag_sums = df.drop(['clip_id', "mp3_path"], axis=1).sum().sort_values(ascending=False)
sorted_tags = tag_sums.index.tolist()
sorted_tags_50 = sorted_tags[:50]
sorted_tags_50plus = sorted_tags[50:]

# concatenate clip_id with df[sorted_tags]
df_clip_sorted = pd.concat([df["clip_id"], df[sorted_tags]], axis=1)
df_clip_sorted = pd.concat([df["mp3_path"], df_clip_sorted], axis=1)
top50_song_identifiers = np.load("./data/top50identifiers.npy")
# top50_song_identifiers = np.load("top50identifiers.npy")
top50_mp3_fp = np.array([x.split("\t")[1] for x in top50_song_identifiers])


sorted_tags_50plus.insert(0, "clip_id")
sorted_tags_50plus.insert(0, "mp3_path")
fifty_plus = df_clip_sorted[sorted_tags_50plus]
fifty_plus = fifty_plus[~fifty_plus["mp3_path"].isin(top50_mp3_fp)]

fifty_plus_multilabel = fifty_plus[fifty_plus.iloc[:, 2:].sum(axis=1) > 1]

fifty_plus_multilabel_tags_sorted = fifty_plus_multilabel.iloc[:,2:].sum(axis=0).sort_values(ascending=False)


# top 50 songs
sorted_tags_50.insert(0, "clip_id")
sorted_tags_50.insert(0, "mp3_path")
topfifty = df_clip_sorted[sorted_tags_50]
t50_train = np.load(T50_TRAIN)
t50_train = np.array([x.split("\t")[1] for x in t50_train])
t50_valid = np.load(T50_VALID)
t50_valid = np.array([x.split("\t")[1] for x in t50_valid])
t50_test = np.load(T50_TEST)
t50_test = np.array([x.split("\t")[1] for x in t50_test])


def get_n_class_k_shot(n, k, df, tags, next_n):
    rng = np.random.RandomState(42)
    next_n_classes = tags[:next_n]

    next_n_classes = rng.choice(next_n_classes, n, replace=False).tolist()
    
    subset = pd.DataFrame()
    remaining_set = df
    for cl in next_n_classes:
        cl_subset = remaining_set[remaining_set[cl] == 1].sample(n=k, replace=False)
        subset = pd.concat([subset, cl_subset])
        remaining_set = remaining_set[~remaining_set["clip_id"].isin(cl_subset["clip_id"])]
    assert subset.shape[0] == n*k
    next_n_classes.insert(0, "clip_id")
    next_n_classes.insert(0, "mp3_path")
    return subset[next_n_classes], next_n_classes[2:]

def get_n_class_k_shot_subset(n, k):
    
    return get_n_class_k_shot(n, k, fifty_plus_multilabel, fifty_plus_multilabel_tags_sorted.index, 10)

def get_n_class_all_subset(n):
    # guarantee that the tags are the same
    _, n_classes_from_k_shot = get_n_class_k_shot_subset(n, k=2)
    n_class_all_data = fifty_plus_multilabel[fifty_plus_multilabel[n_classes_from_k_shot].sum(axis=1) >0]
    n_classes_from_k_shot.insert(0, "clip_id")
    n_classes_from_k_shot.insert(0, "mp3_path")
    return n_class_all_data[n_classes_from_k_shot], n_classes_from_k_shot[2:]


def get_n_class_k_shot_top50(n,k):
    t50_train_df = topfifty[topfifty["mp3_path"].isin(t50_train)]
    
    return get_n_class_k_shot(n, k, t50_train_df, sorted_tags_50[2:], 50)

def get_eval_data_top50(n, k, valid_or_test):
    _, n_classes_from_k_shot = get_n_class_k_shot_top50(n, k)
    
    if valid_or_test == "valid":
        t50_eval_df = topfifty[topfifty["mp3_path"].isin(t50_valid)]
    else:
        t50_eval_df = topfifty[topfifty["mp3_path"].isin(t50_test)]
    
    n_classes_eval_data = t50_eval_df[t50_eval_df[n_classes_from_k_shot].sum(axis=1) >0]
    n_classes_from_k_shot.insert(0, "clip_id")
    n_classes_from_k_shot.insert(0, "mp3_path")
    return n_classes_eval_data[n_classes_from_k_shot], n_classes_from_k_shot[2:]
    
    
    
            
    
if __name__ == "__main__":
    five_way_two_shot, tags = get_n_class_k_shot_top50(4, 1)
    five_way_two_shot_eval, tags_eval = get_eval_data_top50(3, 10, "valid")
    breakpoint()

    