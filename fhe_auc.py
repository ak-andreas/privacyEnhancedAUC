import numpy as np
from Pyfhel import Pyfhel
import random
import time
from sklearn import metrics
import matplotlib.pyplot as plt


# Create HE object
n_mults = 5
HE = Pyfhel(
    key_gen=True,
    context_params={
        "scheme": "CKKS",
        "n": 2**15,
        "scale": 2**30,
        "qi_sizes": [60] + [30] * n_mults + [60],
    },
)

HE.keyGen()
HE.relinKeyGen()


# calculates the ground truth AUC with sklearn that we compare
# our results to
def get_true_auc(full_labels, full_pred_scores):
    y = np.array(full_labels)
    pred = np.array(full_pred_scores)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    truth = metrics.auc(fpr, tpr)
    return truth

# load data
def file_to_list(path, dataset_size="100"):
    data = []
    f = open(path, "r")
    for lines in f:
        line = lines.rstrip()
        data.append(float(line))
    return data


def create_threshold_to_TP_FP_vector(scores, labels, num_threshold: int):
    thresholds = np.linspace(0, 1.0, num=num_threshold, endpoint=True)[::-1]
    epsilon = 0
    tp_vector = [0] * (len(thresholds))
    fp_vector = [0] * (len(thresholds))
    for i in range(len(thresholds)):
        for j in range(len(labels)):
            if scores[j] + epsilon >= thresholds[i]:
                if labels[j] == 1.0:
                    tp_vector[i] += 1
                else:
                    fp_vector[i] += 1

    return np.array(tp_vector, dtype=np.float64), np.array(fp_vector, dtype=np.float64)


def accumulate_TP_FP_vectors(all_clients_data: list, num_threshold=10):
    accumulated_data = []

    for client_data in all_clients_data:
        labels, scores = zip(*client_data)
        TP_vector, FP_vector = create_threshold_to_TP_FP_vector(
            scores, labels, num_threshold
        )

        TP_vector_shifted = np.zeros_like(TP_vector)
        TP_vector_shifted[1:] = TP_vector[:-1]
        FP_vector_shifted = np.zeros_like(FP_vector)
        FP_vector_shifted[1:] = FP_vector[:-1]
        TP_vector_enc = HE.encryptFrac(TP_vector)
        FP_vector_enc = HE.encryptFrac(FP_vector)
        TP_vector_shifted_enc = HE.encryptFrac(TP_vector_shifted)
        FP_vector_shifted_enc = HE.encryptFrac(FP_vector_shifted)
        accumulated_data.append(
            [TP_vector_enc, TP_vector_shifted_enc, FP_vector_enc, FP_vector_shifted_enc]
        )

    return accumulated_data


def split_data_among_users(full_labels, full_pred_scores, number_of_clients=1):

    # pack labels and scores together
    paired_data = list(zip(full_labels, full_pred_scores))

    # shuffle the lines 
    random.shuffle(paired_data)

    # split among clients
    split_size = len(paired_data) // number_of_clients
    clients_data = [
        paired_data[i * split_size : (i + 1) * split_size]
        for i in range(number_of_clients)
    ]

    # If there are remaining items, distribute them to clients
    remaining = len(paired_data) % number_of_clients
    for i in range(remaining):
        clients_data[i].append(paired_data[-(i + 1)])
    clients_data = [
        sorted(client, key=lambda x: x[1], reverse=True) for client in clients_data
    ]
    return clients_data


def server_computation(data, vector_length=10):
    TP_sum = HE.encryptFrac(np.array([0] * vector_length, dtype=np.float64))
    TP_sum_shifted = HE.encryptFrac(np.array([0] * vector_length, dtype=np.float64))
    FP_sum = HE.encryptFrac(np.array([0] * vector_length, dtype=np.float64))
    FP_sum_shifted = HE.encryptFrac(np.array([0] * vector_length, dtype=np.float64))
    for d in data:
        TP_sum += d[0]
        TP_sum_shifted += d[1]
        FP_sum += d[2]
        FP_sum_shifted += d[3]

    # create a mask to fetch TP_N and FP_N
    vector = [0] * vector_length  
    vector[-1] = 1
    mask_enc = HE.encryptFrac(np.array(vector, dtype=np.float64))
    TP_N = ~(TP_sum * mask_enc)
    HE.rescale_to_next(TP_N)
    TP_N = HE.cumul_add(TP_N)
    FP_N = ~(FP_sum * mask_enc)
    HE.rescale_to_next(FP_N)
    FP_N = HE.cumul_add(FP_N)

    two_enc = HE.encryptFrac(np.array([2], dtype=np.float64))

    den = ~(TP_N * FP_N)
    HE.rescale_to_next(den)
    den = ~(two_enc * den)

    num = ~((TP_sum + TP_sum_shifted) * (FP_sum - FP_sum_shifted))
    num = HE.cumul_add(num)

    # todo mult with random value
    p = random.randint(2, 99)
    return num * p, den * p


def client_computation(num, den):
    result_num = HE.decryptFrac(num)
    result_den = HE.decryptFrac(den)
    return result_num[0] / result_den[0]


def plot_threshold_AUC(clients_data, truth, thresholds):
    aucs = []
    for t in thresholds:
        print(f"started with threshold {t}")
        accumulated_data = accumulate_TP_FP_vectors(clients_data, num_threshold=t)
        result_num_enc, result_den_enc = server_computation(
            accumulated_data, vector_length=t
        )
        result_num = HE.decryptFrac(result_num_enc)
        result_den = HE.decryptFrac(result_den_enc)
        res = result_num[0] / result_den[0]
        print(res)
        aucs.append(res)

    # Plotting the AUC vs Thresholds
    plt.figure(figsize=(8, 5))
    plt.axhline(y=truth, color="r", linestyle="--", label="Ground Truth AUC")
    plt.plot(thresholds, aucs, marker="o")
    plt.title("AUC vs amount of threshold")
    plt.xlabel("thresholds")
    plt.ylabel("AUC")
    plt.grid(True)
    plt.savefig("thresholds_vs_auc.png", format="png")


def main():

    full_labels_100 = file_to_list("data/labels_100.txt")
    full_pred_scores_100 = file_to_list("data/pred_cons_100.txt")
    true_auc_100 = get_true_auc(full_labels_100, full_pred_scores_100)
    clients_data_100 = split_data_among_users(full_labels_100, full_pred_scores_100)
    accumulated_data_100 = accumulate_TP_FP_vectors(clients_data_100, 100)

    full_labels_1000 = file_to_list("data/labels_1000.txt")
    full_pred_scores_1000 = file_to_list("data/pred_cons_1000.txt")
    true_auc_1000 = get_true_auc(full_labels_1000, full_pred_scores_1000)
    clients_data_1000 = split_data_among_users(full_labels_1000, full_pred_scores_1000)
    accumulated_data_1000 = accumulate_TP_FP_vectors(clients_data_1000, 100)

    full_labels_10000 = file_to_list("data/labels_10000.txt")
    full_pred_scores_10000 = file_to_list("data/pred_cons_10000.txt")
    true_auc_10000 = get_true_auc(full_labels_10000, full_pred_scores_10000)
    clients_data_10000 = split_data_among_users(
        full_labels_10000, full_pred_scores_10000
    )
    accumulated_data_10000 = accumulate_TP_FP_vectors(clients_data_10000, 100)

    full_labels_100000 = file_to_list("data/labels_100000.txt")
    full_pred_scores_100000 = file_to_list("data/pred_cons_100000.txt")
    true_auc_100000 = get_true_auc(full_labels_100000, full_pred_scores_100000)
    clients_data_100000 = split_data_among_users(
        full_labels_100000, full_pred_scores_100000
    )
    accumulated_data_100000 = accumulate_TP_FP_vectors(clients_data_100000, 100)

    # Time measurement for sample size 100

    # EXPERIMENT 1: Time vs sample size

    print("EXPERIMENT 1: Time vs sample size")

    start_time = time.time()
    result_num_enc_100, result_den_enc_100 = server_computation(
        accumulated_data_100, 100
    )
    client_result_100 = client_computation(result_num_enc_100, result_den_enc_100)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ground true for 100 samples: {true_auc_100}")
    print(client_result_100)
    print(f"Elapsed time for 100 samples: {elapsed_time:.4f} seconds")

    start_time = time.time()
    result_num_enc_100, result_den_enc_100 = server_computation(
        accumulated_data_100, 100
    )
    client_result_100 = client_computation(result_num_enc_100, result_den_enc_100)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ground true for 100 samples: {true_auc_100}")
    print(client_result_100)
    print(f"Elapsed time for 100 samples: {elapsed_time:.4f} seconds")

    # Time measurement for sample size 1000
    start_time = time.time()
    result_num_enc_1000, result_den_enc_1000 = server_computation(
        accumulated_data_1000, 100
    )
    client_result_1000 = client_computation(result_num_enc_1000, result_den_enc_1000)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ground true for 1000 samples: {true_auc_1000}")
    print(client_result_1000)
    print(f"Elapsed time for 1000 samples: {elapsed_time:.4f} seconds")

    # Time measurement for sample size 10000
    start_time = time.time()
    result_num_enc_10000, result_den_enc_10000 = server_computation(
        accumulated_data_10000, 100
    )
    client_result_10000 = client_computation(result_num_enc_10000, result_den_enc_10000)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ground true for 10000 samples: {true_auc_10000}")
    print(client_result_10000)
    print(f"Elapsed time for 10000 samples: {elapsed_time:.4f} seconds")

    # Time measurement for sample size 10000
    start_time = time.time()
    result_num_enc_100000, result_den_enc_100000 = server_computation(
        accumulated_data_100000, 100
    )
    client_result_100000 = client_computation(
        result_num_enc_100000, result_den_enc_100000
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ground true for 100000 samples: {true_auc_100000}")
    print(client_result_100000)
    print(f"Elapsed time for 100000 samples: {elapsed_time:.4f} seconds")

    print("EXPERIMENT 2: Time vs number of clients")

    full_labels = file_to_list("data/labels_100000.txt")
    full_pred_scores = file_to_list("data/pred_cons_100000.txt")
    true_auc = get_true_auc(full_labels, full_pred_scores)
    clients_data_1_client = split_data_among_users(full_labels, full_pred_scores, 1)
    accumulated_data_1_client = accumulate_TP_FP_vectors(clients_data_1_client, 100)

    clients_data_3_client = split_data_among_users(full_labels, full_pred_scores, 3)
    accumulated_data_3_client = accumulate_TP_FP_vectors(clients_data_3_client, 100)

    clients_data_5_client = split_data_among_users(full_labels, full_pred_scores, 5)
    accumulated_data_5_client = accumulate_TP_FP_vectors(clients_data_5_client, 100)

    clients_data_10_client = split_data_among_users(full_labels, full_pred_scores, 10)
    accumulated_data_10_client = accumulate_TP_FP_vectors(clients_data_10_client, 100)

    clients_data_25_client = split_data_among_users(full_labels, full_pred_scores, 25)
    accumulated_data_25_client = accumulate_TP_FP_vectors(clients_data_25_client, 100)

    clients_data_50_client = split_data_among_users(full_labels, full_pred_scores, 50)
    accumulated_data_50_client = accumulate_TP_FP_vectors(clients_data_50_client, 100)

    # Time measurement for  1client
    start_time = time.time()
    result_num_enc, result_den_enc = server_computation(accumulated_data_1_client, 100)
    client_result = client_computation(result_num_enc, result_den_enc)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ground true for 100000 samples: {true_auc_100000}")
    print(client_result)
    print(f"Elapsed time for 1 client: {elapsed_time:.4f} seconds")

    start_time = time.time()
    result_num_enc, result_den_enc = server_computation(accumulated_data_1_client, 100)
    client_result = client_computation(result_num_enc, result_den_enc)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ground true for 100000 samples: {true_auc_100000}")
    print(client_result)
    print(f"Elapsed time for 1 client: {elapsed_time:.4f} seconds")

    # Time measurement for client
    start_time = time.time()
    result_num_enc, result_den_enc = server_computation(accumulated_data_3_client, 100)
    client_result = client_computation(result_num_enc, result_den_enc)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ground true for 100000 samples: {true_auc_100000}")
    print(client_result)
    print(f"Elapsed time for 3 client: {elapsed_time:.4f} seconds")

    # Time measurement for client
    start_time = time.time()
    result_num_enc, result_den_enc = server_computation(accumulated_data_5_client, 100)
    client_result = client_computation(result_num_enc, result_den_enc)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ground true for 100000 samples: {true_auc_100000}")
    print(client_result)
    print(f"Elapsed time for 5 client: {elapsed_time:.4f} seconds")

    # Time measurement for client
    start_time = time.time()
    result_num_enc, result_den_enc = server_computation(accumulated_data_10_client, 100)
    client_result = client_computation(result_num_enc, result_den_enc)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ground true for 100000 samples: {true_auc_100000}")
    print(client_result)
    print(f"Elapsed time for 10 client: {elapsed_time:.4f} seconds")

    # Time measurement for client
    start_time = time.time()
    result_num_enc, result_den_enc = server_computation(accumulated_data_25_client, 100)
    client_result = client_computation(result_num_enc, result_den_enc)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ground true for 100000 samples: {true_auc_100000}")
    print(client_result)
    print(f"Elapsed time for 25 client: {elapsed_time:.4f} seconds")

    # Time measurement for client
    start_time = time.time()
    result_num_enc, result_den_enc = server_computation(accumulated_data_50_client, 100)
    client_result = client_computation(result_num_enc, result_den_enc)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ground true for 100000 samples: {true_auc_100000}")
    print(client_result)
    print(f"Elapsed time for 50 client: {elapsed_time:.4f} seconds")

    # EXPERIMENT 3
    print(true_auc_100000)
    plot_threshold_AUC(
        clients_data_1_client, true_auc_100000, [10, 50, 100, 250, 500, 1000]
    )


if __name__ == "__main__":
    main()
