"""
    hierarchical clustering with custom distance function (here: representations + cosine)
    adapted from: https://gist.github.com/codehacken/8b9316e025beeabb082dda4d0654a6fa

"""
import sys
import os
import argparse
import logging
from typing import List, Dict
import random
import torch
import pandas as pd
import time
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sentence_transformers import SentenceTransformer, util
from global_identifiable import STRANFORMERS_CACHE, set_cache
from global_const import SEED, set_global_seed, set_logging, get_complete_model_name_from_path, get_results_folder, \
    SUBREDDIT_U2_COL, SUBREDDIT_U1_COL, SUBREDDIT_A_COL, CONVERSATION_U2_COL, CONVERSATION_U1_COL, CONVERSATION_A_COL, \
    ID_U2_COL, ID_U1_COL, ID_A_COL, AUTHOR_U2_COL, AUTHOR_U1_COL, AUTHOR_A_COL, U2_COL, U1_COL, ANCHOR_COL


sys.path.append(os.path.join('..', 'style_embed/utility'))
from statistics_utility import mean_confidence_interval

set_cache()
set_logging()

SAMPLE_SIZE = 5000


def main(model_path: str,
         test_file: str,
         sample_size=SAMPLE_SIZE, sim_threshold=0.9, n_clusters=None):
    """
    cluster the representations of a model into n_clusters clusters
    :param model_path: path to trained representation model
    :param test_file: path to CAV tasks from which we sample utterances to cluster
        (we sample from the test file in the paper --> variable is called test_file)
    :param sample_size: # of CAV tasks to sample
    :param sim_threshold: agg clustering variable
    :param n_clusters: number of clusters to cluster sentences into
    :return:
    """

    assert test_file is not None, logging.error('need a base corpus file for clustering ...')
    sentence_lookup, subreddit_lookup, author_lookup, conversation_lookup = get_lookup_dicts(test_file, sample_size)

    output_dir = get_results_folder(model_path=model_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    logging.info("Working with output folder {}".format(output_dir))

    model = SentenceTransformer(model_path, cache_folder=STRANFORMERS_CACHE)
    if model_path == 'roberta-base':
        # Roberta seems to be initialized with max seq length of 514 leading to errors in encoding
        model.max_seq_length = 512
    model_name = get_complete_model_name_from_path(model_path)

    cluster_vectors(model, output_dir, sentence_lookup=sentence_lookup, subreddit_lookup=subreddit_lookup,
                    author_lookup=author_lookup, conversation_lookup=conversation_lookup, model_name=model_name,
                    sim_threshold=sim_threshold, n_clusters=n_clusters)


def get_lookup_dicts(test_file, sample_size=5000):
    """
    get lookup dictionaries for subreddit, authors and conversations
    :param sample_size:
    :param test_file: tsv file to TAV tasks
    :return:
    """
    # read in tsv file
    #   get column names

    task_data = sample_task_data(test_file, sample_size, seed=SEED)

    # Get unique sentences (can repeat across TAV tasks)
    unique_s_ids = set(task_data[ID_A_COL].tolist() + task_data[ID_U1_COL].tolist() + task_data[ID_U2_COL].tolist())
    logging.info(f"unique number of utterances equates to {len(unique_s_ids)}")
    logging.info(f"  first 100 unique utts: {list(unique_s_ids)[:100]}")

    # Generate dictionaries {sentence_id: author_id/conv_id/sub_id}
    sentence_lookup = {}
    subreddit_lookup = {}
    author_lookup = {}
    conversation_lookup = {}
    for s_id in unique_s_ids:
        if s_id in task_data[ID_A_COL].unique():
            sentence_row = task_data[task_data[ID_A_COL] == s_id]
            sentence_lookup[s_id] = sentence_row[ANCHOR_COL].iloc[0]
            subreddit_lookup[s_id] = sentence_row[SUBREDDIT_A_COL].iloc[0]
            author_lookup[s_id] = sentence_row[AUTHOR_A_COL].iloc[0]
            conversation_lookup[s_id] = sentence_row[CONVERSATION_A_COL].iloc[0]
        elif s_id in task_data[ID_U1_COL].unique():
            sentence_row = task_data[task_data[ID_U1_COL] == s_id]
            sentence_lookup[s_id] = sentence_row[U1_COL].iloc[0]
            subreddit_lookup[s_id] = sentence_row[SUBREDDIT_U1_COL].iloc[0]
            author_lookup[s_id] = sentence_row[AUTHOR_U1_COL].iloc[0]
            conversation_lookup[s_id] = sentence_row[CONVERSATION_U1_COL].iloc[0]
        elif s_id in task_data[ID_U2_COL].unique():
            sentence_row = task_data[task_data[ID_U2_COL] == s_id]
            sentence_lookup[s_id] = sentence_row[U2_COL].iloc[0]
            subreddit_lookup[s_id] = sentence_row[SUBREDDIT_U2_COL].iloc[0]
            author_lookup[s_id] = sentence_row[AUTHOR_U2_COL].iloc[0]
            conversation_lookup[s_id] = sentence_row[CONVERSATION_U2_COL].iloc[0]
    # corpus_sentences = set(task_data[ANCHOR_COL].tolist() + task_data[U1_COL].tolist() + task_data[U2_COL].tolist())

    return sentence_lookup, subreddit_lookup, author_lookup, conversation_lookup


def sample_task_data(test_file, sample_size=SAMPLE_SIZE, seed=SEED):
    task_data = pd.read_csv(test_file, sep='\t')
    logging.info(f"Sampling {sample_size} task data from {test_file} with seed {seed}")
    set_global_seed(seed=SEED)
    task_data = task_data.sample(min(sample_size, len(task_data)))
    return task_data


def cluster_vectors(model, output_dir, sentence_lookup, subreddit_lookup=None, author_lookup=None,
                    conversation_lookup=None, model_name='', sample_size=5000, sim_threshold=0.95, n_clusters=None):
    # https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/agglomerative.py
    # Two parameters to tune:
    #   min_cluster_size: Only consider cluster that have at least 25 elements
    #   threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar

    sentence_ids = list(sentence_lookup.keys())
    sentence_strings = [sentence_lookup[s_id] for s_id in sentence_ids]

    logging.info('Calculating cosine similarities ...')
    # if rep_model:  # called for bi-encoders like SBERT
    # Compute cosine similarity scores
    logging.info("Encode the corpus. This might take a while ... ")
    corpus_embeddings = model.encode(sentence_strings, batch_size=64, show_progress_bar=True,
                                     convert_to_tensor=True)
    cos_scores = torch.clamp(util.cos_sim(corpus_embeddings, corpus_embeddings), max=1).cpu()

    logging.info("Start clustering ... ")
    start_time = time.time()

    # Agg. Clustering returns [a, b, a, a, c], where at pos the cluster is given
    clusters = agg_clustering(cos_scores, sim_threshold=sim_threshold, n_clusters=n_clusters)
    # list of clusters, where cluster is represented with sentence ids
    cluster_by_lists = [[s_id
                         for s_id, cl_id in enumerate(clusters) if cl_id == i]
                        for i in range(max(clusters) + 1)]
    cluster_lookup = {s_id: clusters[s_index] for s_index, s_id in enumerate(sentence_ids)}

    logging.info("Clustering done after {:.2f} sec".format(time.time() - start_time))

    dist_matrix = 1 - cos_scores
    from sklearn.metrics import silhouette_score
    logging.info(f"Silhouette score: "
                 f"{silhouette_score(X=dist_matrix, labels=clusters, metric='precomputed')}")

    at_least_two_clusters = [cluster_list for cluster_list in cluster_by_lists if len(cluster_list) > 1]
    logging.info("Size biggest cluster: {}".format(max(len(cluster_list) for cluster_list in cluster_by_lists)))
    logging.info("number of one element clusters: {}".format(len(cluster_by_lists) - len(at_least_two_clusters)))
    logging.info("number of more than one element clusters: {}".format(len(at_least_two_clusters)))

    from itertools import combinations
    same_author_pairs = 0
    sa_same_conv = 0
    sa_same_sub = 0
    for s1, s2 in combinations(sentence_ids, r=2):
        if author_lookup[s1] == author_lookup[s2]:
            same_author_pairs += 1
            if conversation_lookup[s1] == conversation_lookup[s2]:
                sa_same_conv += 1
            if subreddit_lookup[s1] == subreddit_lookup[s2]:
                sa_same_sub += 1

    logging.info(
        "Share of same author pairs that is also same conversation: {} = {}/{}"
            .format(sa_same_conv / same_author_pairs, sa_same_conv, same_author_pairs))
    logging.info("Share of same author pairs that is also same subreddit: {} = {}/{}"
                 .format(sa_same_sub / same_author_pairs, sa_same_sub, same_author_pairs))

    tmp_clusters = [cl_id for cl_id in clusters if len(cluster_by_lists[cl_id]) > 1]
    tmp_s_ids = [s_id for s_id in sentence_ids if len(cluster_by_lists[cluster_lookup[s_id]]) > 1]

    if subreddit_lookup:
        log_cluster_similarity(clusters, sentence_ids, subreddit_lookup, criteria_name='subreddit')
        # also evaluate on 2 element clusters only?
        log_cluster_similarity(tmp_clusters, tmp_s_ids, subreddit_lookup, criteria_name='>1 subreddit')
    if author_lookup:
        log_cluster_similarity(clusters, sentence_ids, author_lookup, criteria_name='author')
        log_cluster_similarity(tmp_clusters, tmp_s_ids, author_lookup, criteria_name='>1 author')
    if conversation_lookup:
        log_cluster_similarity(clusters, sentence_ids, conversation_lookup, criteria_name='conversation')
        log_cluster_similarity(tmp_clusters, tmp_s_ids, conversation_lookup, criteria_name='>1 conversation')

    print_clustering(cluster_by_lists, sentence_strings, cos_scores)

    df_clusters = pd.DataFrame.from_dict({'Cluster {}'.format(i):
                                              [sentence_strings[s_id] for s_id in cluster_by_lists[i]] for i in
                                          range(len(cluster_by_lists))},
                                         orient='index')

    filename = output_dir + '/cluster_model-{}_sample-{}_{}_n-clusters-{}.tsv'.format(model_name, sample_size,
                                                                                      time.time(), n_clusters)
    logging.info('Saving to {} ... '.format(filename))
    if not os.path.exists(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))
    df_clusters.to_csv(filename, sep='\t')


def log_cluster_similarity(clusters: List[int], sample_s_ids: List, criteria_lookup: Dict,
                           criteria_name: str = 'subreddit'):
    """

    :param clusters: list of integers, same length as sample_s_ids, at position i it has the cluster number
        of sentence sample_s_ids[i]
    :param sample_s_ids: list of ids for sentences that are considered for clustering
    :param criteria_lookup: dict: sample_sid -> cluster_number
    :param criteria_name: name of the considered criteria for logging
    :return:
    """
    nbr_combinations, nbr_distinctcluster, nbr_samecluster, total_distinctcriteria, \
    total_samecluster_pairs, total_samecriteria = calc_cluster_sim(clusters, criteria_lookup, sample_s_ids)

    sample_criteria = list(set(criteria_lookup[s_id] for s_id in sample_s_ids))
    same_criteria_cluster = [sample_criteria.index(criteria_lookup[s_id]) for s_id in sample_s_ids]
    # rand score expects 1-dim array with cluster numbers per sentence id
    cluster_similarity = adjusted_rand_score(same_criteria_cluster, clusters)
    logging.info(' Cluster similarity with {} cluster is {}'.format(criteria_name, cluster_similarity))
    logging.info('      Share of same {} pairs that are in the same cluster: {} = {}/{}'
                 .format(criteria_name, nbr_samecluster / total_samecriteria, nbr_samecluster, total_samecriteria))
    # Calculate cluster scores for 100 random combinations (shuffle the "clusters" list)
    rand_values = []
    for i in range(100):
        _, _, tmp_nbr_samecluster, _, _, tmp_total_samecriteria = \
            calc_cluster_sim(random.sample(list(clusters), len(clusters)), criteria_lookup, sample_s_ids)
        rand_values.append(tmp_nbr_samecluster / tmp_total_samecriteria)
    logging.info('      Share of same {} pairs that are in the same cluster'
                 ' when randomly sampled according to cluster size: {}+-{}'
                 .format(criteria_name, np.mean(rand_values), mean_confidence_interval(rand_values)))
    logging.info('      Share of same cluster pairs that fulfill same {}: {} = {}/{}'
                 .format(criteria_name, nbr_samecluster / total_samecluster_pairs, nbr_samecluster,
                         total_samecluster_pairs))
    logging.info('      Share of distinct {} pairs that are in distinct clusters: {} = {}/{}'
                 .format(criteria_name, nbr_distinctcluster / total_distinctcriteria, nbr_distinctcluster,
                         total_distinctcriteria))
    logging.info('Number of same {} pairs: {} out of total number of considered pairs {}'
                 .format(criteria_name, total_samecriteria, nbr_combinations))


def calc_cluster_sim(clusters, criteria_lookup, sample_s_ids):
    """
        calculate the cluster similarities
    """
    # FOR ALL pair combinations ....
    from itertools import combinations
    nbr_combinations = 0
    nbr_samecluster = 0
    total_samecriteria = 0
    nbr_distinctcluster = 0
    total_distinctcriteria = 0
    total_samecluster_pairs = 0
    cluster_lookup = {s_id: clusters[s_index] for s_index, s_id in enumerate(sample_s_ids)}
    for s1, s2 in combinations(sample_s_ids, r=2):
        nbr_combinations += 1
        if criteria_lookup[s1] == criteria_lookup[s2]:
            total_samecriteria += 1
            if cluster_lookup[s1] == cluster_lookup[s2]:
                nbr_samecluster += 1
        else:
            total_distinctcriteria += 1
            if cluster_lookup[s1] != cluster_lookup[s2]:
                nbr_distinctcluster += 1
        if cluster_lookup[s1] == cluster_lookup[s2]:
            total_samecluster_pairs += 1
    return nbr_combinations, nbr_distinctcluster, nbr_samecluster, total_distinctcriteria, total_samecluster_pairs, \
           total_samecriteria


def print_clustering(clusters, corpus_sentences, cos_sim):
    """
        print cluster
    :param clusters:
    :param corpus_sentences:
    :param cos_sim:
    :return:
    """
    # Print for all clusters the top 3 and bottom 3 elements
    clusters.sort(key=len, reverse=True)
    for i, cluster in enumerate(clusters):
        logging.info("Cluster {}, #{} Elements".format(i + 1, len(cluster)))
        for sentence_id in cluster[0:3]:
            logging.info(f"    {corpus_sentences[sentence_id]} [{cos_sim[cluster[0], sentence_id]}]")
        logging.info("    ...")
        for cl_index, sentence_id in enumerate(cluster[-3:]):
            if cl_index > 2:
                logging.info(f"    {corpus_sentences[sentence_id]} [{cos_sim[cluster[0], sentence_id]}]")


def agg_clustering(cos_scores, sim_threshold, n_clusters=None):
    """
        call sklearn AgglomerativeClustering with pre-computed matrix
    """
    # Perform agglomerative clustering.
    # The affinity is precomputed (since the distance are precalculated).
    # Use an 'average' linkage. Use any other apart from  'ward'.
    dist_matrix = 1 - cos_scores
    if sim_threshold == None:
        distance_threshold = None
    else:
        distance_threshold = 1 - sim_threshold
    import sys
    # https://stackoverflow.com/questions/57401033/how-to-fixrecursionerror-maximum-recursion-depth-exceeded-while-getting-the-st
    sys.setrecursionlimit(100000)

    if n_clusters is not None:
        distance_threshold = None

    agg = AgglomerativeClustering(affinity='precomputed', n_clusters=n_clusters,
                                  linkage='complete', distance_threshold=distance_threshold)  # n_clusters=3, 'average'

    # Use the distance matrix directly.
    u = agg.fit_predict(dist_matrix)
    # print(corpus_sentences)
    # print(u)
    return u


if __name__ == "__main__":
    """
        Example call: python cluster.py -md ".../roberta-base-loss-triplet-margin-0.5-evaluator-triplet/seed-106" 
        -test ".../train_data/test-45000__subreddits-100-2018_tasks-300000__topic-variable-conversation.tsv" -n 7
    """

    parser = argparse.ArgumentParser(description='Evaluating a Neural Model.')
    parser.add_argument('-md', '--model_dir', default=None, help='path to trained model, '
                                                                 'do not refer directly to the bin file')
    parser.add_argument('-test', '--test_dataset', default=None, help="path to dataset "
                                                                      "that the triple task should be tested on")
    parser.add_argument('-n', "--n_clusters", default=100, help="nbr of sentences to sample")

    # NOT CALLED IN BATCH FILE
    parser.add_argument('-s', "--sample_nbr", default=SAMPLE_SIZE, help="nbr of sentences to sample")

    args = parser.parse_args()

    main(model_path=args.model_dir, test_file=args.test_dataset, sample_size=int(args.sample_nbr), sim_threshold=None,
         n_clusters=int(args.n_clusters))
