from unittest import TestCase
from global_const import set_logging
import logging
from fine_tune import main

set_logging()


class Test(TestCase):
    def setUp(self) -> None:
        self.cache_folder = "/home/anna/sentence_transformer/"
        self.sub_train = "fixtures/train_data/" \
                         "train-7__subreddits-2_year-2018-2018_tasks-10_topic_variable-subreddit.tsv"
        self.sub_dev = "fixtures/train_data/dev-1__subreddits-2_year-2018-2018_tasks-10_topic_variable-subreddit.tsv"

    def test_main(self):
        logging.info('Starting test ...')

        margin = 0.6

        main(train_file=self.sub_train, dev_file=self.sub_dev, model_key="bert-base-uncased", loss="contrastive",
             margin=margin, cache_folder=self.cache_folder, batch_size=2, eval=False, epochs=1)

