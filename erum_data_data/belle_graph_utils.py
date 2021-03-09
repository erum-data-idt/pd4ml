import numpy as np

"""
Some helper functions for the graph version of the Belle dataset
"""

def adjacency_matrix_from_mothers_np(mother_indices, symmetrize=True, add_diagonal=True):
    """
    Calculate adjacency matrix from mother indices (numpy version). Assumes
    that mother indices are -1 padded.
    """
    N = mother_indices.shape[1]
    adj = np.eye(N + 1, dtype=np.int8)[mother_indices][:, :, :-1]
    if symmetrize:
        adj = adj + np.transpose(adj, (0, 2, 1))
    if add_diagonal:
        adj += np.where(
            (mother_indices != -1)[:, :, np.newaxis],
            np.eye(N, dtype=np.int8),
            0
        )
    adj[adj > 1] = 1
    return adj


def adjacency_matrix_from_mothers_tf(mother_indices, symmetrize=True, add_diagonal=True):
    """
    Like `adjacency_matrix_from_mothers_np`, but using tensorflow
    """
    import tensorflow as tf

    shape = tf.shape(mother_indices)
    N = shape[1]
    bs = shape[0]
    inputs = mother_indices

    idx = tf.where(inputs < 0, tf.cast(N, dtype=tf.int64), inputs)
    adj = tf.one_hot(tf.cast(idx, dtype=tf.int32), N + 1)[:, :, :-1]

    if symmetrize:
        adj = adj + tf.linalg.matrix_transpose(adj)

    if add_diagonal:
        diagonal = tf.broadcast_to(tf.eye(N), (bs, N, N))
        diagonal = tf.where(
            tf.repeat(tf.reshape(inputs != -1, (bs, N, 1)), N, axis=2),
            diagonal,
            tf.zeros_like(adj),
        )
        adj = adj + diagonal

    adj = tf.cast(adj != 0, dtype=tf.float32)
    return adj


def get_sorted_pdg():
    from erum_data_data import Belle
    data = Belle.load()
    mapped_pdg, counts = np.unique(data[0][0][:, :, -1].ravel(), return_counts=True)
    mapped = mapped_pdg[np.argsort(counts)][::-1].astype(int)
    return mapped[mapped != 0]


# generated with `get_sorted_pdg()`
sorted_pdg = [
   506, 423, 236, 143, 321, 171, 323,  15, 114,  19,  96, 394, 372,
    46,  44, 190, 134, 141, 375,   8, 337, 220, 415, 248, 249,  11,
    34, 434, 263,   4, 126, 378, 418, 183, 182,  10, 245,  85,  45,
   210, 178, 139, 275, 322, 419, 121, 437,  42, 118, 424, 480, 483,
    79, 414,  61, 393,  38, 349, 108,  86,  32, 357, 238, 173, 305,
    57, 413, 491, 158, 355,  78, 176, 164, 352, 187, 205, 471,  71,
   458, 350, 101, 280, 389, 376,  89, 120, 457, 436, 186, 444,  76,
   390, 402, 342, 258, 391, 407, 162, 185, 152, 276, 149, 302, 106,
   110, 401, 109, 487, 279, 192, 446, 334,  56, 359, 194, 314, 202,
   420, 208, 427, 475, 495, 465, 442, 489, 360, 448, 216, 215, 300,
   214,  13, 227, 501, 502, 399, 344, 474, 150,  53, 325, 129, 326,
   195, 366, 370, 336, 163, 154,  66, 105, 285, 473, 455, 421,  43,
   119, 271, 373, 363, 428, 304, 229, 425, 133, 270, 219, 479, 112,
   250, 217,  33, 403, 338, 206,  39, 268, 130, 328,  93, 346, 212
]


def remap_pdg(mapped_pdg):
    """
    Remap mapped pdg ids again for later one-hot encoding to `num_pdg`
    number of dimensions. They are sorted by number of occurence.
    """

    remap_dict = {v : k for k, v in enumerate(sorted_pdg)}

    @np.vectorize
    def remap(pdg_id):
        return remap_dict.get(pdg_id, len(sorted_pdg) + 1)

    return remap(mapped_pdg)


def np_onehot(indices, depth):
    """
    Behaviour like tf.one_hot
    """
    indices = np.array(indices)
    indices[indices >= depth] = depth
    return np.eye(depth + 1, dtype=np.int8)[indices][..., :-1]
