from unittest import TestCase
from global_const import set_logging, set_global_seed
from global_identifiable import STRANFORMERS_CACHE
from neural_trainer import SentenceBertFineTuner

set_logging()
set_global_seed()


class SentenceBertFineTunerTest(TestCase):
    def setUp(self) -> None:
        self.conv_train = "fixtures/train_data/" \
                          "train-7__subreddits-2_year-2018-2018_tasks-10_topic_variable-conversation.tsv"
        self.rand_train = "fixtures/train_data/" \
                          "train-210000__subreddits-100-2018_tasks-300000__topic-variable-random.tsv"
        self.rand_dev = "fixtures/train_data/dev-45000__subreddits-100-2018_tasks-300000__topic-variable-random.tsv"
        self.cache_folder = STRANFORMERS_CACHE

    def test_set_SBTuner(self):
        tuner = SentenceBertFineTuner(self.rand_train, self.rand_dev, seed=100, debug=True)

    def test_get_input_examples(self):
        examples = SentenceBertFineTuner.get_input_examples(self.conv_train)
        for i, ex in enumerate(examples):
            if i == 0:
                self.assertEqual(ex.texts[0], "sure, but that\'s not the choice.")
                self.assertEqual(ex.texts[1], "originally, Syracuse was a potential location for Cornell University "
                                              "but Ezra adamantly hated Syracuse so he chose to place the school in "
                                              "Ithaca. How would you feel if Cornell had been in Syracuse instead?\n\n")
            self.assertLessEqual(len(ex.texts[0].split(" ")), 512)
            self.assertLessEqual(len(ex.texts[1].split(" ")), 512)

    def test_batch_rand(self):
        roberta_tuner = SentenceBertFineTuner(self.rand_train, self.rand_dev, model_path="roberta-base",
                                              margin=0.5, loss="triplet",
                                              evaluation_type="triplet", cache_folder=self.cache_folder)
        dataloader = roberta_tuner.train(debug_dataloader=True)

        for i, elem in enumerate(dataloader):
            if i in [11767, 11768]:
                print(elem)


