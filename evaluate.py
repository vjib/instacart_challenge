import pandas as pd
import numpy as np

def calculate_f1_score(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0.
    precision = 1. * cross_size / len(y_pred)
    recall = 1. * cross_size / len(y_true)
    return (2 * precision * recall) / (precision + recall)

K = 5 # Number of order set. For this challenge, we use only 5 different sets
DATASET_PATH = './dataset/'
SUBMISSION_PATH = './submissions/'

test_set = pd.read_csv(DATASET_PATH + 'test_set.csv')
test_dict = dict()

for x in test_set.itertuples():
    test_dict[x.order_id] = x.products.split()

f1_score_means = []
for i in range(K):
    submission_set = pd.read_csv(SUBMISSION_PATH + 'submission_' + str(i+1) + '.csv')
    submission_dict = dict()

    f1_scores = []
    for x in submission_set.itertuples():
        y_true = test_dict[x.order_id]
        y_pred = x.products.split()
        f1_scores.append( calculate_f1_score(y_true, y_pred) )

    f1_score_mean = np.mean(f1_scores)
    f1_score_means.append(f1_score_mean)
    print('#' + str(i+1), 'set:', f1_score_mean)

print('Total Average:', np.mean(f1_score_means))
