"""
    Some code for STEL error analysis (Section 4.2), needs STEL as this is an error analysis on STEL resutls
"""
import pandas as pd
from global_identifiable import include_STEL_project
from global_const import set_logging, set_global_seed
from typing import List
import logging

set_logging()
set_global_seed()
include_STEL_project()
from eval_style_models import LOCAL_STEL_CHAR_QUAD, read_in_stel_instances
from to_add_const import LOCAL_STEL_DIM_QUAD


class ErrorAnalysis:

    def __init__(self, sub_tuned: List[str] = None, conv_tuned: List[str] = None, rand_tuned: List[str] = None,
                 base_paths: List[str] = None, trip_tuned: List[str] = None, con_tuned: List[str] = None):
        self.stel, _ = read_in_stel_instances(stel_dim_tsv=LOCAL_STEL_DIM_QUAD, stel_char_tsv=LOCAL_STEL_CHAR_QUAD,
                                              filter_majority_votes=True)
        self.sub_paths = sub_tuned
        self.rand_paths = rand_tuned
        self.conv_paths = conv_tuned
        self.base_paths = base_paths
        self.trip_paths = trip_tuned
        self.con_paths = con_tuned

    def find_un_learned_loss(self, loss="trip"):
        if loss == "trip":
            tuned_paths = self.trip_paths
        elif loss == "con":
            tuned_paths = self.con_paths
        else:
            return
        self.find_unlearned_stel_instances(self.base_paths, tuned_paths).\
            to_csv(f"../output/{loss}_unlearned.tsv", sep="\t")
        self.find_learned_stel_instances(self.base_paths, tuned_paths).\
            to_csv(f"../output/{loss}_learned.tsv", sep="\t")

    def find_un_learned_topic(self, topic="conv"):
        if topic == "conv":
            tuned_paths = self.conv_paths
        elif topic == "sub":
            tuned_paths = self.sub_paths
        elif topic == "rand":
            tuned_paths = self.rand_paths
        else:
            return
        self.find_unlearned_stel_instances(self.base_paths, tuned_paths).\
            to_csv(f"../output/{topic}_unlearned.tsv", sep="\t")
        self.find_learned_stel_instances(self.base_paths, tuned_paths).\
            to_csv(f"../output/{topic}_learned.tsv", sep="\t")

    def find_unlearned_model(self, model='bert'):
        if model == 'bert':
            base_paths = [self.base_paths[0]]
            tuned_paths = [self.rand_paths[0], self.conv_paths[0], self.sub_paths[0]]
        elif model == 'Bert':
            base_paths = [self.base_paths[1]]
            tuned_paths = [self.rand_paths[1], self.conv_paths[1], self.sub_paths[1]]
        elif model == 'RoBERTa':
            base_paths = [self.base_paths[2]]
            tuned_paths = [self.rand_paths[2], self.conv_paths[2], self.sub_paths[2]]
        logging.info("UNLEARNED: ")
        self.find_unlearned_stel_instances(base_paths, tuned_paths).to_csv(f"../output/{model}_unlearned.tsv", sep="\t")
        logging.info("LEARNED: ")
        self.find_learned_stel_instances(base_paths, tuned_paths).to_csv(f"../output/{model}_learned.tsv", sep="\t")
        # logging.info(f"Statistics unlearned instances: {unlearned_instances['style type'].value_counts()}")
        # self.print_stel_instances(unlearned_instances)
        # return unlearned_instances

    def find_unlearned_all(self):
        self.find_unlearned_stel_instances(self.base_paths, [*self.rand_paths, *self.sub_paths, *self.conv_paths])\
            .to_csv(f"../output/all_R_unlearned.tsv", sep="\t")
        self.find_learned_stel_instances(self.base_paths, [*self.rand_paths, *self.sub_paths, *self.conv_paths])\
            .to_csv(f"../output/all_R_learned.tsv", sep="\t")

    def find_learned_stel_instances(self, base_paths: List[str], tuned_paths: List[str]):
        return self.find_un_learned_stel_instances(base_paths, tuned_paths, unlearned=False)

    def find_unlearned_stel_instances(self, base_paths: List[str], tuned_paths: List[str]):
        return self.find_un_learned_stel_instances(base_paths, tuned_paths, unlearned=True)

    def find_un_learned_stel_instances(self, base_paths: List[str], tuned_paths: List[str], unlearned=True):
        base_dfs = [pd.read_csv(base_path, sep="\t") for base_path in base_paths]
        tuned_dfs = [pd.read_csv(tuned_path, sep="\t") for tuned_path in tuned_paths]

        if unlearned:
            id_list = self.get_unlearned_ids(base_dfs, tuned_dfs)
        else:
            id_list = self.get_learned_ids(base_dfs, tuned_dfs)

        self.unlearned_stel_instances = self.stel[self.stel['ID'].isin(id_list)]

        import statistics_utility
        vote_mean = statistics.mean(
            self.unlearned_stel_instances
            [self.unlearned_stel_instances['style type'] == 'simplicity']
            ['# Votes out of 5 for Correct Alternative'])

        logging.info(f"Statistics unlearned={unlearned} instances: {self.unlearned_stel_instances['style type'].value_counts()}")
        logging.info(f"Average Vote for the GT ordering for simplicty: {vote_mean}")
        self.print_sample_instances(self.unlearned_stel_instances)

        return self.unlearned_stel_instances

    @staticmethod
    def get_learned_ids(base_dfs, tuned_dfs):
        for i, tuned_df in enumerate(tuned_dfs):
            tmp_id_list = tuned_df[tuned_df['Correct Alternative'] == tuned_df['TunedSentenceBertSimilarity']][
                'ID'].to_list()
            if i == 0:
                id_list = set(tmp_id_list)
            else:
                id_list = id_list & set(tmp_id_list)
        for base_df in base_dfs:
            tmp_id_list = base_df[base_df['Correct Alternative'] != base_df['TunedSentenceBertSimilarity']][
                'ID'].to_list()
            id_list = id_list & set(tmp_id_list)
        return id_list

    @staticmethod
    def get_unlearned_ids(base_dfs, tuned_dfs):
        for i, tuned_df in enumerate(tuned_dfs):
            tmp_id_list = tuned_df[tuned_df['Correct Alternative'] != tuned_df['TunedSentenceBertSimilarity']][
                'ID'].to_list()
            if i == 0:
                id_list = set(tmp_id_list)
            else:
                id_list = id_list & set(tmp_id_list)
        for base_df in base_dfs:
            tmp_id_list = base_df[base_df['Correct Alternative'] == base_df['TunedSentenceBertSimilarity']][
                'ID'].to_list()
            id_list = id_list & set(tmp_id_list)
        return id_list

    def print_stel_instance(self, stel_instance):
        logging.info(f"    Category - {stel_instance['style type']}")
        logging.info(f"    A1 - {stel_instance['Anchor 1']}")
        logging.info(f"    A2 - {stel_instance['Anchor 2']}")
        logging.info(f"    S1 - {stel_instance['Alternative 1.1']}")
        logging.info(f"    S2 - {stel_instance['Alternative 1.2']}")
        logging.info(f"    S2 - {stel_instance['ID']}")
        logging.info(f"    # Votes - {stel_instance['# Votes out of 5 for Correct Alternative']}")
        logging.info(f"    Correct Answer - {stel_instance['Correct Alternative'] - 1}")

    def print_sample_instances(self, stel_instances: pd.DataFrame):
        import numpy as np
        np.random.seed(100)
        for stel_dim in pd.unique(stel_instances['style type']):
            sample = stel_instances[stel_instances['style type'] == stel_dim]\
                .sample(min(2,  len(stel_instances[stel_instances['style type'] == stel_dim])))
            self.print_stel_instances(sample)

    def print_stel_instances(self, stel_instances):
        for row_id, stel_instance in stel_instances.iterrows():
            self.print_stel_instance(stel_instance)
