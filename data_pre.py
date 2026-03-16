import os
import numpy as np
from tqdm import tqdm
from scipy import sparse

def build_sparse_matrix():
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


if __name__ == "__main__":
    rating_matrix = build_sparse_matrix()
    sparse.save_npz("netflix_matrix.npz", rating_matrix)