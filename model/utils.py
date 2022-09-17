import rle


def reset_index_by_time(clusterd_idxs: list):
    """[kmeansなどで与えられた時系列と無関係なindexを出てきた番号から振り直す]

    Args:
        clusterd_idxs (list): [description]

    Returns:
        [type]: [description]
    """
    sorted_cluster_idx = []
    cluster_dict = {}
    for c in clusterd_idxs:
        if c not in cluster_dict:
            cluster_dict[c] = len(cluster_dict)
        sorted_cluster_idx.append(cluster_dict[c])
    return sorted_cluster_idx


def run_length_encode(input_seq):
    rle_label_list = []
    labels, durs = rle.encode(input_seq)
    for label_i, dur_i in zip(labels, durs):
        rle_label_list.extend([label_i+str(dur_i)] * dur_i)
    assert len(input_seq) == len(rle_label_list)
    return rle_label_list
