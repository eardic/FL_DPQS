import argparse
import copy
import glob
import os
import os.path as osp
import pickle
import sys
from matplotlib.cbook import maxdict

import matplotlib.pyplot as plt
import matplotlib.style as mpls
import numpy as np
from tqdm import tqdm

mpls.use('fast')
plt.rcParams["figure.figsize"] = (20, 10)  # w,h
plt.rcParams.update({'font.size': 16})


def dval_btw_rounds(client_dict, first_cli_ix, out_dir, begin=0, end=30):
    plt.clf()
    c_data = client_dict[first_cli_ix]
    c_data_rnds, c_data_vals = c_data['rounds'], c_data["vals"]
    r_sorted_ix = np.argsort(c_data_rnds)
    r_end_ix = r_sorted_ix[-1]
    for i in r_sorted_ix:
        if c_data_rnds[i] >= end:
            r_end_ix = i
            break
    r_start_ix = r_sorted_ix[0]
    for i in r_sorted_ix:
        if c_data_rnds[i] >= begin:
            r_start_ix = i
            break
    print(f"Drawing values of client {first_cli_ix} between rounds "
          f"({c_data_rnds[r_start_ix]} and {c_data_rnds[r_end_ix]})...")
    colors = None
    if "labels" in c_data:
        colors = ['red' if l == 1 else 'green' for l in c_data["labels"]]
    plt.scatter(c_data_vals[r_start_ix], c_data_vals[r_end_ix], c=colors)
    plt.xlabel(f"Data Value (R={c_data_rnds[r_start_ix]})")
    plt.ylabel(f"Data Value (R={c_data_rnds[r_end_ix]})")
    plt.savefig(osp.join(out_dir,
                         f"by_btw_rnds_{c_data_rnds[r_start_ix]}_{c_data_rnds[r_end_ix]}_c_{first_cli_ix}.png"),
                dpi=600)


def dval_by_data(client_dict, out_dir):
    data_vals = []  # data count x data values
    data_colors = []
    for c_ix, c_data in client_dict.items():
        vals = c_data["vals"]
        vals = np.array(vals).T.tolist()
        data_vals.extend(vals)
        if "labels" in c_data:
            colors = ['red' if l == 1 else 'green' for l in c_data["labels"]]
            data_colors.extend(colors)
    mean_data_vals = []
    for ix, vals in enumerate(data_vals):
        avg = np.mean(vals)
        mean_data_vals.append(avg)
    mean_data_vals = np.array(mean_data_vals, copy=False)
    if len(data_colors) > 0:
        sum_dict = {"good": 0, "good_count": 0, "bad": 0, "bad_count": 0}
        good_ixs, bad_ixs = [], []
        for ix, (vals, color) in enumerate(zip(data_vals, data_colors)):
            vals = np.array(vals)
            if color == "green":
                sum_dict["good"] += np.sum(vals)
                sum_dict["good_count"] += len(vals.flatten())
                good_ixs.append(ix)
            else:
                sum_dict["bad"] += np.sum(vals)
                sum_dict["bad_count"] += len(vals.flatten())
                bad_ixs.append(ix)

        plt.rcParams["figure.figsize"] = (10, 10)  # w,h

        print("Drawing hist values by mean data values...")
        plt.clf()
        plt.hist(mean_data_vals[good_ixs], bins=100, color="green", alpha=0.5)
        plt.hist(mean_data_vals[bad_ixs], bins=100, color="red", alpha=0.5)
        plt.xlabel("Data Value")
        plt.ylabel("Count")
        plt.savefig(osp.join(out_dir, "hist_by_dataval.png"), dpi=300)

        avg_good = sum_dict["good"] / (sum_dict["good_count"] + sys.float_info.epsilon)
        avg_bad = sum_dict["bad"] / (sum_dict["bad_count"] + sys.float_info.epsilon)
        old_fig_size = plt.rcParams["figure.figsize"]
        print("Drawing values by data type...")
        plt.clf()
        plt.bar(["noisy", "clean"], [avg_bad, avg_good], width=0.4, color=["red", "green"], align="center")
        plt.text("noisy", avg_bad, f"{avg_bad:.6f}")
        plt.text("clean", avg_good, f"{avg_good:.6f}")
        plt.xlabel("Data Type")
        plt.ylabel("Average Data Value")
        plt.savefig(osp.join(out_dir, "by_data_type.png"), dpi=300)

        plt.rcParams["figure.figsize"] = old_fig_size
    print("Drawing values by data index...")
    plt.clf()
    plt.scatter(np.arange(len(mean_data_vals)),
                mean_data_vals, s=1,
                c=data_colors if len(data_colors) > 0 else None)
    plt.xlabel("Data Index")
    plt.ylabel("Data Value")
    plt.savefig(osp.join(out_dir, "by_data_ix.png"), dpi=600)


def dval_by_client_idx(client_dict, out_dir):
    print("Drawing values by client index...")
    plt.clf()
    for c_ix, c_data in client_dict.items():
        vals = c_data["vals"]
        vals = np.array(vals).mean(axis=0).flatten()
        colors = None
        if "labels" in c_data:
            colors = ['red' if l == 1 else 'green' for l in c_data["labels"]]
        plt.scatter([c_ix] * len(vals), vals, s=1, c=colors)
    plt.xlabel("Client Index")
    plt.ylabel("Data Value")
    plt.savefig(osp.join(out_dir, "by_client_ix.png"), dpi=600)

    print("Drawing line chart for 1 noise and 3 clean samples...")
    plt.clf()
    cix, c_data = next(iter(client_dict.items()))
    print("Selected client: ", cix)
    if "labels" in c_data:
        lbls = np.array(c_data["labels"], copy=False)
        vals = np.array(c_data["vals"], copy=False)
        round_ixs = np.array(c_data["rounds"], copy=False)
        sorted_rounds = np.argsort(round_ixs)
        round_ixs = round_ixs[sorted_rounds]
        vals = vals[sorted_rounds]
        noisy_ix = np.argwhere(lbls == 1).flatten()[-1]
        clean_ixs = np.argwhere(lbls == 0).flatten()[:3]
        plt.plot(round_ixs, vals[:, noisy_ix], color='red')
        for i in clean_ixs:
            plt.plot(round_ixs, vals[:, i], '--', color='green')
        plt.xlabel("Round Index")
        plt.ylabel("Data Value")
        plt.savefig(osp.join(out_dir, "dval_line_by_client.png"), dpi=600)

    print("Drawing stats by client index...")
    plt.clf()
    g_count_arr = []
    b_count_arr = []
    count_arr = []
    max_v, min_v, mean_v = 0, sys.float_info.max, 0
    for c_ix, c_data in client_dict.items():
        total_count = len(c_data["vals"][-1])
        count_arr.append(total_count)
        if "labels" in c_data:
            g_count, b_count = 0, 0
            for l in c_data["labels"]:
                if l == 1:
                    b_count += 1
                else:
                    g_count += 1
            g_count_arr.append(g_count)
            b_count_arr.append(b_count)
        max_v = max(max_v, total_count)
        min_v = min(min_v, total_count)
        mean_v += total_count
    cl_arr = list(client_dict.keys())
    mean_v = int(mean_v / len(cl_arr))
    if len(g_count_arr) > 0 and len(g_count_arr) == len(b_count_arr):
        plt.bar(cl_arr, b_count_arr, color="red")
        plt.bar(cl_arr, g_count_arr, bottom=b_count_arr, color="green")
    else:
        plt.bar(cl_arr, count_arr)
    plt.xlabel("Client Index")
    plt.ylabel("Data Size")
    plt.title(f"Max: {max_v}, Min:{min_v}, Mean: {mean_v}")
    plt.savefig(osp.join(out_dir, "stats_by_client_ix.png"), dpi=600)


def dval_by_round(out_dir, round_dict):
    # print("Drawing values by round index...")
    # plt.clf()
    # for r_ix, r_data in round_dict.items():
    #     vals = r_data["vals"]
    #     colors = None
    #     if "labels" in r_data:
    #         colors = ['red' if l == 1 else 'green' for l in r_data["labels"]]
    #     plt.scatter([r_ix] * len(vals), vals, s=1, c=colors)
    # plt.xlabel("Round Index")
    # plt.ylabel("Data Value")
    # plt.savefig(osp.join(out_dir, "by_round_ix.png"), dpi=600)

    print("Drawing normalized values by round index...")
    plt.clf()
    for r_ix, r_data in round_dict.items():
        vals = np.array(r_data["vals"], copy=False)
        colors = None
        if "labels" in r_data:
            colors = np.array(['red' if l == 1 else 'green' for l in r_data["labels"]], copy=False)
        
        filter = np.argsort(vals)[:-int(len(vals) * 0.05)].flatten().tolist()
        colors = colors[filter]
        vals = vals[filter]
        
        min, max = np.min(vals), np.max(vals)
        vals = (vals - min) / (max-min)

        cnt = np.sum(vals > 0.5)
        
        plt.scatter(r_ix, cnt, s=10, marker="x", c="black")
        # plt.scatter([r_ix] * len(vals), vals, s=1, c=colors)

    plt.xlabel("Round Index")
    plt.ylabel("Data Value")
    plt.savefig(osp.join(out_dir, "by_round_ix_scaled.png"), dpi=600)

    # print("Drawing stats by round index...")
    # plt.clf()
    # g_count_arr = []
    # b_count_arr = []
    # count_arr = []
    # max_v, min_v, mean_v = 0, sys.float_info.max, 0
    # for r_ix, r_data in round_dict.items():
    #     total_count = len(r_data["vals"])
    #     count_arr.append(total_count)
    #     if "labels" in r_data:
    #         g_count, b_count = 0, 0
    #         for l in r_data["labels"]:
    #             if l == 1:
    #                 b_count += 1
    #             else:
    #                 g_count += 1
    #         g_count_arr.append(g_count)
    #         b_count_arr.append(b_count)
    #     max_v = max(max_v, total_count)
    #     min_v = min(min_v, total_count)
    #     mean_v += total_count
    # rnd_arr = list(round_dict.keys())
    # mean_v = int(mean_v / len(rnd_arr))
    # if len(g_count_arr) > 0 and len(g_count_arr) == len(b_count_arr):
    #     plt.bar(rnd_arr, b_count_arr, color="red")
    #     plt.bar(rnd_arr, g_count_arr, bottom=b_count_arr, color="green")
    # else:
    #     plt.bar(rnd_arr, count_arr)
    # plt.xlabel("Round Index")
    # plt.ylabel("Data Size")
    # plt.title(f"Max: {max_v}, Min:{min_v}, Mean: {mean_v}")
    # plt.savefig(osp.join(out_dir, "stats_by_round_ix.png"), dpi=600)


def report_dir(args):
    out_dir = os.path.join("reports", os.path.split(args.input)[1])
    os.makedirs(out_dir, exist_ok=True)

    # MNIST'te user name key olarak yazılıyor bunun client index olarak değiştirilmesi gerek !!!
    client_noise_data = {}
    if args.noise_data is not None:
        with open(args.noise_data, "rb") as f:
            client_noise_data = pickle.load(f)
            print(f"Loaded noise data with {len(client_noise_data)} clients !")

    file_iter = glob.glob(os.path.join(args.input, "**/*.npy"))
    client_dict = {}
    round_dict = {}
    first_cli_ix = None
    for file in tqdm(file_iter, total=len(file_iter), desc="Loading..."):
        round_root, file_name = os.path.split(file)
        root, round_name = os.path.split(round_root)
        round_ix = int(round_name.split("_")[1])
        client_ix = int(os.path.splitext(file_name)[0])
        data_vals = np.load(file)

        client_info = client_dict.setdefault(client_ix,
                                             {"vals": [], "labels": [], "rounds": []})
        client_info["vals"].append(data_vals)
        client_info["rounds"].append(round_ix)

        round_info = round_dict.setdefault(round_ix,
                                           {"vals": [], "labels": []})
        round_info["vals"].extend(data_vals.tolist())

        noise_idxs = client_noise_data.get(client_ix, None)
        if noise_idxs is not None:
            lbls = np.zeros(data_vals.shape[0], dtype=np.uint8)
            lbls[noise_idxs] = 1
            lbls = lbls.tolist()
            client_info["labels"] = lbls
            round_info["labels"].extend(lbls)

        if round_ix == 0 and first_cli_ix is None:
            first_cli_ix = client_ix
            print("Pivot client index: ", client_ix)

    plt.ioff()

    # if first_cli_ix is not None:
    #     dval_btw_rounds(client_dict, first_cli_ix, out_dir, 0, 30)
    #     dval_btw_rounds(client_dict, first_cli_ix, out_dir, 30, 60)
    #     dval_btw_rounds(client_dict, first_cli_ix, out_dir, 60, 90)

    # dval_by_client_idx(client_dict, out_dir)
    dval_by_round(out_dir, round_dict)
    # dval_by_data(client_dict, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        type=str,
                        # default="cifar_data_val/",
                        # default="data_val/mnist-1000-100-e5-allb-fbgnorm-n0.6-20221023-005650",
                        default="data_val/cifar10-1k-100-e5-allb-fbgnorm-step0.5-clean-20221113-114304",
                        help="The path of dir containing data values grouped by rounds.")
    parser.add_argument("--mode", default="single", choices=["multi", "single"])
    parser.add_argument("--noise_data",
                        default="fedcv_data/cifar10_client_noisy_data_idxs_1000.pickle",
                        # default="fedcv_data/mnist_client_noisy_data_idxs_1000.pickle",
                        help="Path of pickle file containing noisy data indexes per user")
    args = parser.parse_args()

    if args.mode == "multi":
        for p_dir in os.listdir(args.input):
            args_c = copy.deepcopy(args)
            args_c.input = os.path.join(args.input, p_dir)
            print("Reporting directory: ", args_c.input)
            report_dir(args_c)
    else:
        print("Reporting single directory: ", args.input)
        report_dir(args)

    print("Done !")
