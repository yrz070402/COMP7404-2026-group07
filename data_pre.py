import os
import numpy as np
from tqdm import tqdm
from scipy import sparse

def build_sparse_matrix_netflix():
    # data path
    floder_path = r"./netflix_data"
    # data file name
    file_names = [f"combined_data_{x}.txt" for x in range(1,5)]

    row_idx = []
    col_idx = []
    ratings = []

    user_id_map = {}
    user_index = 0
    movie_id = 0
    user_idx = 0

    for file in file_names:
        path = os.path.join(floder_path, file)

        with open(path, 'r') as f:
            for line in tqdm(f):
                line = line.strip()
                # For last empty line
                if not line:
                    break
                # Process the movie ID line and the record line separately.
                if line.endswith(":"):
                    num_str = line[:-1]
                    movie_id = int(num_str) - 1
                else:
                    record = line.split(',')
                    user_id = int(record[0])
                    rating = float(record[1])
                    # User ids are not contiguous. When finding a new id, give it a new index
                    if user_id_map.get(user_id) is None:
                        user_id_map[user_id] = user_index
                        user_index += 1
                    
                    user_idx = user_id_map[user_id]

                    row_idx.append(user_idx)
                    col_idx.append(movie_id)
                    ratings.append(rating)
    rating_matrix = sparse.csr_matrix((ratings, (row_idx, col_idx)), shape=(user_index, movie_id+1))

    return rating_matrix

def build_sparse_matrix_movielens():
    floder_path = r"./movielens_data"
    file_name = r"ratings.dat"
    sep = "::"
    path = os.path.join(floder_path, file_name)
    users, items, ratings = [], [], []
    with open(path, 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            u, it, r, _ts = line.split(sep)
            users.append(int(u) - 1)     # IDs 从 1 开始
            items.append(int(it) - 1)
            ratings.append(float(r))

    users = np.array(users, dtype=np.int32)
    items = np.array(items, dtype=np.int32)
    ratings = np.array(ratings, dtype=np.float32)
    n_users = int(users.max()) + 1
    n_items = int(items.max()) + 1

    rating_matrix = sparse.coo_matrix((ratings, (users, items)), shape=(n_users, n_items), dtype=np.float32).tocsr()
    return rating_matrix


if __name__ == "__main__":
    rating_matrix_movielens = build_sparse_matrix_movielens()
    sparse.save_npz("movielens_matrix.npz", rating_matrix_movielens)
    rating_matrix_netflix = build_sparse_matrix_netflix()
    sparse.save_npz("netflix_matrix.npz", rating_matrix_netflix)