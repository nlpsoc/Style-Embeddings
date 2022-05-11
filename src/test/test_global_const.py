from unittest import TestCase
from global_identifiable import TRAIN_DATA_BASE
from global_const import HPC_DEV_DATASETS, generate_file_prefix, get_results_folder


class TestGenerateFile(TestCase):
    def setUp(self) -> None:
        self.model_path = "/hpc/uu_cs_nlpsoc/02-awegmann/sentence_transformersav-models/topic-sub/bert-base-uncased-loss-triplet-margin-0.5-evaluator-triplet"
        self.model_path_w_seed = "/hpc/uu_cs_nlpsoc/02-awegmann/sentence_transformersav-models/topic-sub/" \
                                 "bert-base-uncased-loss-triplet-margin-0.5-evaluator-triplet/seed-104"
        self.triple_task_filename = HPC_DEV_DATASETS[0]

    def test_generate_file_prefix(self):
        file_prefix = generate_file_prefix(self.model_path, self.triple_task_filename)
        self.assertEqual(file_prefix, f"dev-45000__subreddits-100-2018_tasks-300000__topic-variable-conversation_"
                                      f"topic-sub-bert-base-uncased-loss-triplet-margin-0.5-evaluator-triplet")

    def test_results_folder(self):
        results_folder = get_results_folder(self.model_path)
        self.assertEqual(results_folder,
                         f"/hpc/uu_cs_nlpsoc/02-awegmann/style_discovery/results/topic-sub/bert/triplet/margin-0.5")

        results_folder = get_results_folder(self.model_path_w_seed)
        self.assertEqual(results_folder,
                         f"/hpc/uu_cs_nlpsoc/02-awegmann/style_discovery/results/topic-sub/bert/triplet/margin-0.5/"
                         f"seed-104")


