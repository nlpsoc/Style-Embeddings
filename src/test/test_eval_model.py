from unittest import TestCase
import logging

from sentence_transformers import SentenceTransformer

import eval_model
import global_const
from eval_model import _evaluate_model
from global_const import set_logging
from trained_similarities import TunedSentenceBertSimilarity

global_const.set_logging()
# from global_identifiable import include_STEL_project
# include_STEL_project()


class Test(TestCase):
    def setUp(self) -> None:
        self.base_model = "roberta-base"  # "paraphrase-MiniLM-L3-v2"  # smallest best model sbert, during experiments: "roberta-base"

        # fixtures with made-up train data
        self.base_dir = "fixtures/train_data/"
        self.dev_conv_org = 'dev-45000__subreddits-100-2018_tasks-300000__topic-variable-conversation.tsv'
        self.sub_dev = self.base_dir + "dev-1__subreddits-2_year-2018-2018_tasks-10_topic_variable-subreddit.tsv"
        self.rand_dev = self.base_dir + "dev-1__subreddits-2_year-2018-2018_tasks-10_topic_variable-random.tsv"
        self.conv_dev = self.base_dir + "dev-1__subreddits-2_year-2018-2018_tasks-10_topic_variable-conversation.tsv"

        self.conv_train = self.base_dir + \
                          "train-7__subreddits-2_year-2018-2018_tasks-10_topic_variable-conversation.tsv"
        self.rand_train = self.base_dir + "train-7__subreddits-2_year-2018-2018_tasks-10_topic_variable-random.tsv"

        self.error = "../style_embed/test-1__subreddits-1_year-2018-2018_tasks-10_topic_variable-conversation.tsv"
        # STEL proprietary data include (comment if not present)
        import os
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.full_STEL_DIM = [cur_dir + '/../Data/STEL/dimensions/_quad_stel-dimensions_formal-815_complex-815.tsv']

    def test_main_wo_STEL(self):
        # test eval run for a base model and fixed small train data in fixtures/train_data
        global_const.set_logging()
        logging.info("Starting to test eval model..")
        logging.info("Starting to test eval model..")
        eval_model.main(model_path=self.base_model,
                        test_files=[self.error], test_stel=False)

    def test_base_model_w_STEL(self):
        # also run on STEL
        eval_model.main(model_path=self.base_model, test_files=[self.conv_train, self.rand_train], test_stel=True)

    def test_wo_STEL_orgmodel(self):
        from sentence_transformers import SentenceTransformer
        from utility.trained_similarities import TunedSentenceBertSimilarity
        from eval_model import _evaluate_model
        from global_const import set_logging
        set_logging()

        model_path = "/home/anna/Documents/UU/Style-Embeddings/Data/Models/AV-Domain_seed-105"
        # test_path = "/home/anna/Documents/UU/Style-Embeddings/Data/train_data/" \
        #             "test-45000__subreddits-100-2018_tasks-300000__topic-variable-conversation.tsv"
        test_path = "fixtures/train_data/train-7__subreddits-2_year-2018-2018_tasks-10_topic_variable-conversation.tsv"

        model = SentenceTransformer('AnnaWegmann/Style-Embedding')

        model_w_sim = TunedSentenceBertSimilarity(model=model)
        _evaluate_model(model_w_sim, model_path, [test_path], test_stel=False, test_AV=True)