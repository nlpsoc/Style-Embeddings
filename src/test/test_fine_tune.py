from unittest import TestCase
from global_const import set_logging
import logging
from fine_tune import main
from global_identifiable import STRANFORMERS_CACHE

set_logging()


class Test(TestCase):
    def setUp(self) -> None:
        # set your own cache location in global_identifiable or here for testing
        self.cache_folder = STRANFORMERS_CACHE
        self.sub_train = "fixtures/train_data/" \
                         "train-7__subreddits-2_year-2018-2018_tasks-10_topic_variable-conversation.tsv"
        self.sub_dev = "fixtures/train_data/dev-1__subreddits-2_year-2018-2018_tasks-10_topic_variable-conversation.tsv"

    def test_main(self):
        logging.info('Starting test ...')

        margin = 0.5
        triplet = "triplet"

        # RoBERTa base with triplet loss was best-performing model
        main(train_file=self.sub_train, dev_file=self.sub_dev, model_key="roberta-base", loss=triplet,
             margin=margin, cache_folder=self.cache_folder, batch_size=4, eval=False, epochs=1,
             evaluation_type=triplet, profile=False)

