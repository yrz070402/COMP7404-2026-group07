import os
import pic_plot
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse

'''
    PureSVD procedure:
    each row of U denotes the characteristic vector of an user
    each row of V denotes the characteristic vector for an item
    u_i*v_j can denotes to the score of probablity user i will like movie j
'''
def pure_svd(matrix_path, f):
    # Assume that rating_matrix has the dimension n_users*n_items
    rating_matrix = sparse.load_npz(matrix_path).astype(float)
    # Then the dimension of W is n_users*f and V is n_items*f
    W, sigma, Vt = svds(rating_matrix, k=f)
    U = W @ np.diag(sigma)
    V = Vt.T
    return U, V


'''
    Top-T inner product
'''
def topT(U, V, users, T):
    v_t = V.T
    users_tops = {}
    for i in users:
        scores = U[i] @ v_t
        top_idx = np.argpartition(-scores, T)[:T]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        users_tops[i] = top_idx
    return users_tops


'''
    L2LSH: baseline function
    Using Class to implement batch computation of K clusters of L2LSH hash functions
'''
class L2LSH:
    def __init__(self, dim, K, r, seed):
        self.dim = int(dim)
        self.K = K
        self.r = r
        rng = np.random.default_rng(seed)
        self.A = rng.normal(0.0, 1.0, size=(self.K, self.dim)).astype(np.float32)
        self.b = rng.uniform(0.0, self.r, size=(self.K, )).astype(np.float32)
    
    def hash_matrix(self, X):
        # X is (N, dim) and A is (K, dim), the project result is N items finishing K different hashing processes
        product = X @ self.A.T
        hash_result = np.floor((product + self.b[None, :]) / self.r)
        return hash_result


'''
    ALSH: Asymmetric Locality Sensitive Hashing
    two function: Q(x) and P(x)
'''
def query_transform(U, m):
    # assume that query q is normalized before using Q(x)
    norm_U = np.linalg.norm(U, axis=1, keepdims=True)
    U_normalized = U / (norm_U + 1e-9)

    # calculate the padding of Q
    Q_padding = np.full((U.shape[0], m), 0.5)
    Q = np.hstack([U_normalized, Q_padding])

    return Q, U_normalized

def preprocess_transform(V, U_param, m):
    # scaling all element by the same constant does not change the result of c-NN
    max_norm_V = np.max(np.linalg.norm(V, axis=1))
    V_scaled = V / (max_norm_V / U_param)

    # calculate the padding of P
    V_scaled_sq = np.sum(V_scaled**2, axis=1, keepdims=True)
    pad_list = []
    for i in range(m):
        pad_list.append(2**i)
    pad_arr = np.array(pad_list)
    P_padding = V_scaled_sq ** pad_arr
    P = np.hstack([V_scaled, P_padding])

    return P, V_scaled
    

'''
    Metric calculation:
    Precision: relevant seen divided by k
    Recall: relevant seen divided by T
'''
def matches(u_hash, v_hashes):
    matches = np.sum(v_hashes == u_hash[None, :], axis=1)
    return matches

def metrix_calculate(ranking, tops_T, k, T):
    precision = []
    recall = []
    hit = 0
    for i in range(k):
        if ranking[i] in tops_T:
           hit += 1
        precision.append(hit/(i+1))
        recall.append(hit/T)
    ps = np.array(precision)
    rs = np.array(recall)
    return ps, rs

def average_pr(pr_list):
    P = np.mean(np.stack([p for p, r in pr_list], axis=0), axis=0)
    R = np.mean(np.stack([r for p, r in pr_list], axis=0), axis=0)
    return P, R

'''
    Main Experiment
    There two parts in the experiment
    The PartI shows that ALSH is worked as expected (the func experiment)
    The PartII shows that the r_alsh chosen in PartI is reasonable (the func experiment4r and def get_avg_pr4r)
'''
def experiment(U, V, Q, P, r_alsh, K_list, T_list, r_list, num_user, random_seed):
    rng = np.random.default_rng(random_seed)
    chosen_users = rng.choice(U.shape[0], size=num_user, replace=False)
    
    # calculate the top-T inner product items for each chosen user
    T_max = max(T_list)
    users_tops = topT(U, V, chosen_users, T_max)

    # Calculate precision and recall for alsh with fixed r
    pr_alsh = {}
    for K in K_list:
        alsh = L2LSH(dim=P.shape[1], K=K, r=r_alsh, seed=random_seed)
        p_hash = alsh.hash_matrix(P)
        
        pr_byT = {}
        for T in T_list:
            prs = []
            for i in tqdm(chosen_users, desc=f"ALSH K={K}, T={T}"):
                q_hash = alsh.hash_matrix(Q[i:i+1])[0]
                m_sum = matches(q_hash, p_hash)
                ranking = np.argsort(-m_sum)
                tops_T = set(users_tops[i][:T].tolist())
                p, r = metrix_calculate(ranking, tops_T, len(ranking), T)
                prs.append((p, r))

            p_avg, r_avg = average_pr(prs)
            pr_byT[T] = (p_avg, r_avg)
        pr_alsh[K] = pr_byT

    # Calculate precision and recall for l2lsh
    pr_baseline = {}
    for K in K_list:
        pr_baseline[K] = {}
        for r0 in r_list:
            l2lsh = L2LSH(dim=V.shape[1], K=K, r=r0, seed=random_seed)
            v_hash = l2lsh.hash_matrix(V)

            pr_byT = {}
            for T in T_list:
                prs = []
                for i in tqdm(chosen_users, desc=f"L2LSH K={K}, T={T}, r={r0}"):
                    u_hash = l2lsh.hash_matrix(U[i:i+1])[0]
                    m_sum = matches(u_hash, v_hash)
                    ranking = np.argsort(-m_sum)
                    tops_T = set(users_tops[i][:T].tolist())
                    p, r = metrix_calculate(ranking, tops_T, len(ranking), T)
                    prs.append((p, r))
                p_avg, r_avg = average_pr(prs)
                pr_byT[T] = (p_avg, r_avg)
            pr_baseline[K][r0] = pr_byT
    
    return pr_alsh, pr_baseline

def experiment4r(U, V, Q, P, num_user, K, T_list, r_list, random_seed):
    rng = np.random.default_rng(random_seed)
    chosen_users = rng.choice(U.shape[0], size=num_user, replace=False)

    # calculate the top-T inner product items for each chosen user
    T_max = max(T_list)
    users_tops = topT(U, V, chosen_users, T_max)

    # Calculate precision and recall for alsh with fixed r
    pr_alsh = {}
    for r0 in r_list:
        alsh = L2LSH(dim=P.shape[1], K=K, r=r0, seed=random_seed)
        p_hash = alsh.hash_matrix(P)
        
        pr_byT = {}
        for T in T_list:
            prs = []
            for i in tqdm(chosen_users, desc=f"ALSH4r K={K}, T={T}, r={r0}"):
                q_hash = alsh.hash_matrix(Q[i:i+1])[0]
                m_sum = matches(q_hash, p_hash)
                ranking = np.argsort(-m_sum)
                tops_T = set(users_tops[i][:T].tolist())

                p, r = metrix_calculate(ranking, tops_T, len(ranking), T)
                prs.append((p, r))

            p_avg, r_avg = average_pr(prs)
            pr_byT[T] = (p_avg, r_avg)
        pr_alsh[r0] = pr_byT
    return pr_alsh

def get_avg_pr4r(U, V, Q, P, num_user, K, T_list, r_list, base_seed, n_runs=5):
    all_result = []

    for run_idx in range(n_runs):
        print(f"Run {run_idx+1}/{n_runs}")
        seed_now = base_seed + run_idx * 7404

        current_result = experiment4r(
            U=U,
            V=V,
            Q=Q, 
            P=P, 
            num_user=num_user, 
            K=K, 
            T_list=T_list, 
            r_list=r_list, 
            random_seed=seed_now
        )
        all_result.append(current_result)

    avg_result = {}
    for r0 in r_list:
        avg_result[r0] = {}
        for T in T_list:
            P_curves = [all_result[i][r0][T][0] for i in range(n_runs)]
            R_curves = [all_result[i][r0][T][1] for i in range(n_runs)]

            P_avg = np.mean(P_curves, axis=0)
            R_avg = np.mean(R_curves, axis=0)

            avg_result[r0][T] = (P_avg, R_avg)
    
    return avg_result


'''
    Command-line argument reading (not necessary for every param has a given defualt)
'''
def parse_args():
    parser = argparse.ArgumentParser(description="Run ALSH")

    parser.add_argument('--db', nargs='?', default='Netflix',
	                    help='Input database name')
    parser.add_argument('--input', nargs='?', default='netflix_matrix.npz',
	                    help='Input csr matrix path')
    parser.add_argument('--f_param', type=int, default=300,
	                    help='latent dimension when doing PureSVD procedure')
    parser.add_argument('--U_param', type=float, default=0.83,
	                    help='U parameter')
    parser.add_argument('--padding', type=int, default=3,
	                    help='padding length')
    parser.add_argument('--num_users', type=int, default=2000,
	                    help='the number of chosen users')
    parser.add_argument('--num_r_runs', type=int, default=5,
	                    help='the number of times we run alsh experiment to choose best r')
    return parser.parse_args()


if __name__ == "__main__":
    # Parameter given
    args = parse_args()
    # the latent dimension used for given database
    fs = {
        "Movielens":150,
        "Netflix":300
    }
    f = args.f_param
    U_param = args.U_param
    m = args.padding
    num_user = args.num_users
    r_alsh = 2.5
    K_list = [512, 256, 128, 64]
    T_list = [1, 5, 10]
    r_list = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    random_seed = 42
    matrix_path = args.input
    data_name = args.db
    n_runs = args.num_r_runs

    # PureSVD procedure
    print('PureSVD')
    # U, V = pure_svd(matrix_path, f)

    data = np.load("svd_cache_f300.npz")
    U, V = data['U_vecs'], data['V_vecs']

    # ALSH
    print('ALSH')
    Q, U_normalized = query_transform(U, m)
    P, V_scaled = preprocess_transform(V, U_param, m)

    # Experiment
    # PratI
    print('Experiment')
    pr_alsh, pr_baseline = experiment(
        U=U_normalized,
        V=V_scaled, 
        Q=Q, 
        P=P, 
        r_alsh=r_alsh, 
        K_list=K_list,
        T_list=T_list,
        r_list=r_list,
        num_user=num_user,
        random_seed=random_seed
    )

    fig = pic_plot.plot_pr_curve(pr_alsh, pr_baseline, K_list, T_list, data_name)
    plt.savefig(f"fig_ALSH_{data_name}.png", dpi=300)
    plt.show()

    # PartII
    avg_r_result = get_avg_pr4r(
        U=U_normalized,
        V=V_scaled, 
        Q=Q, 
        P=P, 
        num_user=num_user, 
        K=512, 
        T_list=T_list, 
        r_list=r_list, 
        base_seed=random_seed, 
        n_runs=n_runs
    )

    fig = pic_plot.plot_avg_pr4r(avg_r_result, r_list, T_list, data_name)
    plt.tight_layout()
    plt.savefig(f"fig_r_compare_{data_name}.png", dpi=300)
    plt.show()