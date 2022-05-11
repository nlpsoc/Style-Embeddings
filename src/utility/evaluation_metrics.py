"""
    calculate performance metrics for the (contrastive) AV tasks
"""
import logging
import pickle
import time

import numpy as np
import sklearn

import pandas as pd

from global_identifiable import OUTPUT_FOLDER
from global_const import generate_file_prefix, SAME_AUTHOR_AU1_COL, U2_COL, U1_COL, ANCHOR_COL
from plot import plot_sim_values


def calc_triple_acc_score(sim_function_callable, triple_task_filename, is_t=False):
    """
    Calculates the F1-scores for correctly selecting the same style utterance v or w to original u
     Triples are of the form (u,v,w)
    :param sim_function_callable: a similarity function callable which takes two strings as input and returns a value
    between 0,1 or -1,1, SAME is defined to be at value 1
    :param is_val: if true, the triples have an additional "val_info" metadata element which is returned as val_class
    :param test_filename: triples of form  (same, same, distinct)
    :return:
    """

    task_data = pd.read_csv(triple_task_filename, sep='\t')
    pair_1s = [anchor for anchor in task_data[ANCHOR_COL].tolist() for _ in range(2)]
    pair_2s = combine_lists_in_alternation(task_data[U1_COL].tolist(), task_data[U2_COL].tolist())
    # triple task predicts u1 (i.e., label 0) or u2 (i.e., label 1),
    #   taken from same_author_au1 which is 1 if u1 is correct
    triple_true = [1 - same_au1 for same_au1 in task_data[SAME_AUTHOR_AU1_COL].tolist()]
    av_true = combine_lists_in_alternation(task_data[SAME_AUTHOR_AU1_COL].tolist(),
                                           triple_true)
    input_triple_strings = [row[ANCHOR_COL] + " [SEP] " + row[U1_COL] + " [SEP] " + row[U2_COL]
                            if row[SAME_AUTHOR_AU1_COL] == 1 else
                            row[ANCHOR_COL] + " [SEP] " + row[U2_COL] + " [SEP] " + row[U1_COL]
                            for row_id, row in task_data.iterrows()]

    # CALLING SIMILARITY functions between every pair
    sims = sim_function_callable(pair_1s, pair_2s)

    # AUTHORSHIP VERIFICATION TASK or BINARY TASK
    #   Area under Curve -> i.e., independent from threshold
    #       see also: https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-pythons
    av_auc = sklearn.metrics.roc_auc_score(av_true, sims)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(av_true, sims)
    optimal_thresh_id = np.argmax(tpr - fpr)
    optimal_thresh = thresholds[optimal_thresh_id]

    #   prediction with optimal threshold
    av_prediction = [1 if sim >= optimal_thresh else 0 for sim in sims]
    av_acc = sklearn.metrics.accuracy_score(av_true, av_prediction)
    av_f1 = sklearn.metrics.f1_score(av_true, av_prediction)
    av_precision = sklearn.metrics.precision_score(av_true, av_prediction)
    av_recall = sklearn.metrics.recall_score(av_true, av_prediction)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(av_true, av_prediction).ravel()
    av_specifity = tn / (tn + fp)

    #   prediction with fixed threshold: 0.2 for triple, 0.63 for AV
    if is_t:
        fixed_thresh = 0.2
    else:
        fixed_thresh = 0.6
    av_fixed_prediction = [1 if sim >= fixed_thresh else 0 for sim in sims]
    av_fixed_acc = sklearn.metrics.accuracy_score(av_true, av_fixed_prediction)
    av_fixed_recall = sklearn.metrics.recall_score(av_true, av_fixed_prediction)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(av_true, av_fixed_prediction).ravel()
    av_fixed_specifity = tn / (tn + fp)

    # TRIPLE TASK
    triple_pred = [0 if sim_au1 > sim_au2 else 1 for sim_au1, sim_au2 in zip(sims[::2], sims[1::2])]
    same_sims = [sim_au1 if task_answer == 0 else sim_au2
                 for task_answer, sim_au1, sim_au2 in zip(triple_true, sims[::2], sims[1::2])]
    distinct_sims = [sim_au2 if task_answer == 0 else sim_au1
                     for task_answer, sim_au1, sim_au2 in zip(triple_true, sims[::2], sims[1::2])]

    acc_score = sklearn.metrics.accuracy_score(triple_true, triple_pred)

    result_dict = {
        "acc_score": acc_score,
        "av_thresh": optimal_thresh,
        "av_auc": av_auc,
        "av_acc": av_acc,
        "av_f1": av_f1,
        "av_precision": av_precision,
        "av_recall": av_recall,
        "av_specificity": av_specifity,
        "av_f_recall": av_fixed_recall,
        "av_f_specificity": av_fixed_specifity,
        "av_f_thresh": fixed_thresh,
        "av_sims": sims,
        "av_f_pred": av_fixed_prediction,
        "distinct_sims": distinct_sims,
        "same_sims": same_sims,
        "input_triple_strings": input_triple_strings,
        "triple_pred": triple_pred,
        "triple_true": triple_true,
        "filename": triple_task_filename,
        "av_pred": av_prediction,
        "av_true": av_true
        # "same_author_true": triple_true
    }

    return result_dict


def combine_lists_in_alternation(list1, list2):
    pair_2s = [None] * (len(list1) + len(list2))
    pair_2s[::2] = list1
    pair_2s[1::2] = list2
    return pair_2s


TOP_N_TO_PRINT = 10


def triple_test_sim_function(similarity_function_callable, triple_task_filename, print_top_n: int = TOP_N_TO_PRINT,
                             output_folder=OUTPUT_FOLDER, sim_function_name="", model_prefix="static"):
    """

    :param similarity_function_callable: a function that can call similarities on two list of strings
    :param triple_tasks_filename:
    :param print_top_n:
    :param output_folder:
    :param sim_function_name:
    :param is_val:
    :param model_prefix:
    :param triple_task_filename:
    :return:
    """
    # CALCULATING PERFORMANCE SCORES
    # acc_score, distinct_sims, same_sims, triplets, y_pred, val_class \
    file_base = generate_file_prefix(sim_function_name, triple_task_filename)
    result_dict = calc_triple_acc_score(sim_function_callable=similarity_function_callable,
                                        triple_task_filename=triple_task_filename, is_t="loss-triplet" in file_base)
    # print("triple acc score is at {}".format(result_dict["acc_score"]))
    logging.info("triple acc score is at {}".format(result_dict["acc_score"]))
    # logging.info("  triple f1 score is at {}".format(result_dict["trip_f1"]))
    # logging.info("  triple precision score is at {}".format(result_dict["trip_precision"]))
    # logging.info("  triple recall score is at {}".format(result_dict["trip_recall"]))

    logging.info("AV auc is at {}".format(result_dict["av_auc"]))
    logging.info("  With a threshold of {}: ".format(result_dict["av_thresh"]))
    logging.info("      AV acc score is at {}".format(result_dict["av_acc"]))
    logging.info("      av f1 score is at {}".format(result_dict["av_f1"]))
    logging.info("      av precision score is at {}".format(result_dict["av_precision"]))
    logging.info("      av recall=sensitivity score is at {}, which equals accuracy for (A, SA) pairs"
                 .format(result_dict["av_recall"]))
    logging.info("      av specificity score is at {}, which equals accuracy for (A, DA) pairs"
                 .format(result_dict["av_specificity"]))
    logging.info(f"      av with fixed thresh={result_dict['av_f_thresh']} leads to: ")
    logging.info(f"            av recall is at {result_dict['av_f_recall']} leads to: ")
    logging.info(f"            av specificity is at {result_dict['av_f_specificity']} leads to: ")

    result_dict_name = f"{output_folder}/{file_base}_result-dict_{time.time()}.pickle"
    pickle.dump(result_dict, open(result_dict_name, 'wb'))
    logging.info(f"saved result dict to {result_dict_name}")

    # PLOTTING FOR ILLUSTRATION
    plot_diff_filebase = f"{output_folder}/diff_ACC-{result_dict['acc_score']}_{file_base}_{time.time()}"
    plot_sims_filebase = f"{output_folder}/02_sims_ACC-{result_dict['acc_score']}_{file_base}_{time.time()}"

    logging.info('Going to save plots to {} and{}'.format(plot_diff_filebase, plot_sims_filebase))

    diff_values = plot_from_resultdict(plot_diff_filebase, plot_sims_filebase, result_dict)

    # SAVING single predictions
    pred_save_path = f"{output_folder}/03_TT-AV_single-pred_{file_base}_{time.time()}.tsv"
    double_tt_pred = [t_pred for pair in zip(result_dict["triple_pred"], result_dict["triple_pred"]) for t_pred in pair]
    double_tt_true = [t_pred for pair in zip(result_dict["triple_true"], result_dict["triple_true"]) for t_pred in pair]
    pd.DataFrame(zip(double_tt_pred, double_tt_true, result_dict["av_pred"], result_dict["av_true"],
                     result_dict["av_sims"],
                     [result_dict["av_thresh"] for _ in range(len(double_tt_pred))],
                     [result_dict["av_f_thresh"] for _ in range(len(double_tt_pred))],
                     result_dict["av_f_pred"]),
                 columns=["TT predictions", "TT true", "AV predictions", "AV true", "AV sims", "AV opt thresh",
                          "AV fixed thresh", "AV fixed predictions"]) \
        .to_csv(pred_save_path, sep="\t")
    logging.info(f"Saved single predictions dataframe to {pred_save_path}")

    # LOGGING single results
    logging.info("Looking at top print_top_n=" + str(print_top_n) + " sentences ...")
    #   triples with the GREATEST DIFFERENCE values (i.e., correctly classified)
    logging.info("Top print_top_n diff values: " + str(diff_values[np.argsort(diff_values)[-print_top_n:]]))
    for sequence in list(pd.Series(result_dict["input_triple_strings"])
                         [np.argsort(result_dict["input_triple_strings"])[-print_top_n:]]):
        logging.info("      " + sequence.replace("\n", ""))
    #   triples with the SMALLEST DIFFERENCE values (i.e., negative and incorrectly classified)
    logging.info("Down print_top_n sim diff values: " + str(diff_values[np.argsort(diff_values)[:print_top_n]]))
    for sequence in list(pd.Series(result_dict["input_triple_strings"])[np.argsort(diff_values)[:print_top_n]]):
        logging.info("      " + sequence.replace("\n", ""))
    return result_dict


def plot_from_resultdict(plot_diff_filebase, plot_sims_filebase, result_dict):
    #   PLOT difference values, i.e., the same - distinct similarities, which should always be > 0
    diff_values = np.subtract(result_dict["same_sims"], result_dict["distinct_sims"])
    logging.info('plotting diff values ...')
    # plot_diff_values(diff_values, result_dict["triple_pred"], result_dict["val_class"], plot_diff_filebase)
    #   PLOT similarity values, i.e., same author values should be greater than distinct
    logging.info('plotting sim values ...')
    plot_sim_values(result_dict["same_sims"], result_dict["distinct_sims"], plot_sims_filebase)
    return diff_values
