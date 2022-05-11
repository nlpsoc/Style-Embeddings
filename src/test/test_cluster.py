from unittest import TestCase
from global_const import set_global_seed, set_logging
import cluster

set_logging()
set_global_seed()


class Test(TestCase):
    def setUp(self) -> None:
        # local variables
        self.base_model = "roberta-base"
        self.conv_test = "../../Data/train_data/" \
                         "test-45000__subreddits-100-2018_tasks-300000__topic-variable-conversation.tsv"

    def test_cluster_base(self):
        cluster.main(model_path=self.base_model, test_file=self.conv_test, sample_size=5, sim_threshold=0.6,
                     n_clusters=2)

    def test_get_lookup_dicts(self):
        sent_lookup, sub_lookup, a_lookup, c_lookup = cluster.get_lookup_dicts(self.conv_test)
        # sentence ids are unique
        self.assertEqual(len(sent_lookup), len(set(list(sent_lookup.keys()))))
        # sentence ids are keys for all 3 lookups dicts
        self.assertEqual(sent_lookup.keys(), sub_lookup.keys())
        self.assertEqual(sent_lookup.keys(), a_lookup.keys())
        self.assertEqual(sent_lookup.keys(), c_lookup.keys())
