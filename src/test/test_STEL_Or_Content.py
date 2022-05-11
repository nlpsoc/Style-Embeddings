from unittest import TestCase
import sys, os
from global_identifiable import include_STEL_project

include_STEL_project()
from eval_style_models import read_in_stel_instances, LOCAL_STEL_CHAR_QUAD, LOCAL_STEL_DIM_QUAD

sys.path.append(os.path.join('..', 'utility'))
import STEL_Or_Content


class Test(TestCase):
    def test_content_confuse_stel(self):
        from set_for_global import ANCHOR1_COL, CORRECT_ALTERNATIVE_COL, ALTERNATIVE11_COL, ALTERNATIVE12_COL, \
            ANCHOR2_COL
        stel_instances, _ = read_in_stel_instances(stel_dim_tsv=LOCAL_STEL_DIM_QUAD, stel_char_tsv=LOCAL_STEL_CHAR_QUAD,
                                                   filter_majority_votes=True)
        confused_stel_instances = STEL_Or_Content.get_STEL_Or_Content_from_STEL(stel_instances)
        stel_instances, _ = read_in_stel_instances(stel_dim_tsv=LOCAL_STEL_DIM_QUAD, stel_char_tsv=LOCAL_STEL_CHAR_QUAD,
                                                   filter_majority_votes=True)

        for (_, row), (_, crow) in zip(stel_instances.iterrows(), confused_stel_instances.iterrows()):
            self.assertEqual(row[ANCHOR1_COL], crow[ANCHOR1_COL])
            self.assertIn(row[CORRECT_ALTERNATIVE_COL], [1, 2])
            if row[CORRECT_ALTERNATIVE_COL] == 1:
                self.assertEqual(row[ANCHOR2_COL], crow[ALTERNATIVE12_COL])
                self.assertEqual(row[ALTERNATIVE11_COL], crow[ALTERNATIVE11_COL])
            else:
                self.assertEqual(row[ANCHOR2_COL], crow[ALTERNATIVE11_COL])
                self.assertEqual(row[ALTERNATIVE12_COL], crow[ALTERNATIVE12_COL])
