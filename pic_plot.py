from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

def plot_pr_curve(pr_alsh, pr_baseline, K_list, T_list, data_name):
    # trying to show the same picture the paper offered
    y_max_all = {
        "Movielens":{
            (512, 1): 25, (512, 5): 60, (512, 10): 80,
            (256, 1): 15, (256, 5): 50, (256, 10): 60,
            (128, 1): 15, (128, 5): 40, (128, 10): 50,
            (64,  1): 8,  (64,  5): 25, (64,  10): 30,
        },
        "Netflix":{
            (512, 1): 20, (512, 5): 40, (512, 10): 50,
            (256, 1): 15, (256, 5): 40, (256, 10): 50,
            (128, 1): 8, (128, 5): 25, (128, 10): 30,
            (64,  1): 6,  (64,  5): 15, (64,  10): 20,
        },
    }
    y_max = y_max_all[data_name]

    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.30, hspace=0.25)

    for i, K in enumerate(K_list):
        for j, T in enumerate(T_list):
            ax = axes[i, j]

            for r0, byT in pr_baseline[K].items():
                P, Rr = byT[T]
                ax.plot(Rr * 100, P * 100, linestyle="--", color="black", linewidth=1.0)

            Pp, Rp = pr_alsh[K][T]
            ax.plot(Rp * 100, Pp * 100, linestyle="-", color="red", linewidth=1.2)

            ax.set_xlim(0, 100)
            ax.set_ylim(0, y_max[(K, T)])
            ax.set_xticks([0, 20, 40, 60, 80, 100])

            ymax = y_max[(K, T)]
            if ymax <= 10:
                yt = list(range(0, ymax + 1, 2))
            elif ymax <= 30:
                yt = list(range(0, ymax + 1, 5))
            else:
                yt = list(range(0, ymax + 1, 20 if ymax >= 60 else 10))
            ax.set_yticks(yt)

            ax.grid(True, linestyle=":", linewidth=0.8, color="0.7")
            ax.set_xlabel("Recall (%)")
            ax.set_ylabel("Precision (%)")

            handles = [
                Line2D([0], [0], color="red", lw=1.2, linestyle="-", label="Proposed"),
                Line2D([0], [0], color="black", lw=1.0, linestyle="--", label="L2LSH"),
            ]
            leg = ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=1.0, fancybox=False)
            leg.get_frame().set_edgecolor("black")

            ax.text(0.20, 0.80, data_name, color="red", transform=ax.transAxes,
                    fontsize=8, fontweight="bold")
            ax.text(0.50, 0.40, f"Top {T}, K = {K}", color="black", transform=ax.transAxes,
                    fontsize=8, fontweight="bold")

    return fig


def plot_single(ax, avg_r_result, r_list, T, ymax, data_name):
    for r0 in r_list:
        if r0 in (1, 2.5, 5):
            continue
        P, Rr = avg_r_result[r0][T]
        ax.plot(Rr * 100, P * 100, linestyle="--", color="gray", linewidth=0.8, alpha=0.5)

    P25, R25 = avg_r_result[2.5][T]
    ax.plot(R25 * 100, P25 * 100, linestyle="-", color="red", linewidth=2.5, 
            label="r=2.5 (Recommended)", zorder=10)

    P1, R1 = avg_r_result[1][T]
    ax.plot(R1 * 100, P1 * 100, linestyle="-", color="blue", linewidth=2.0,
            marker="o", markersize=4, markevery=max(1, len(R1)//12), label="r=1", zorder=9)

    P5, R5 = avg_r_result[5][T]
    ax.plot(R5 * 100, P5 * 100, linestyle="-", color="magenta", linewidth=2.0,
            marker="D", markersize=3, markevery=max(1, len(R5)//12), label="r=5", zorder=8)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, ymax)
    ax.set_xticks([0, 20, 40, 60, 80, 100])

    if ymax <= 25:
        ax.set_yticks(list(range(0, ymax + 1, 5)))
    elif ymax <= 60:
        ax.set_yticks(list(range(0, ymax + 1, 10)))
    else:
        ax.set_yticks([0, 20, 40, 60, 80])

    ax.grid(True, linestyle=":", linewidth=0.8, color="0.85", alpha=0.8)
    ax.set_xlabel("Recall (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top-{T} (K=512)", fontsize=12, fontweight="bold")

    ax.legend(loc="upper right", frameon=True, framealpha=0.96, fontsize=10, edgecolor="black")

    ax.text(0.02, 0.98, data_name, color="red", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")


def plot_avg_pr4r(avg_r_result, r_list, T_list, data_name):
    y_max_all = {
        "Movielens":{
            1: 25, 5: 60, 10: 80,
        },
        "Netflix":{
            1: 20, 5: 45, 10: 60,
        },
    }
    y_max = y_max_all[data_name]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.30, hspace=0.35)

    for i, T in enumerate(T_list):
        plot_single(axes[i], avg_r_result, r_list, T, y_max[T], data_name)
    
    fig.suptitle(f"ALSH: Sensitivity to parameter r (m=3, U=0.83, K=512)", 
                 fontsize=13, fontweight="bold", y=1.02)
    
    return fig


