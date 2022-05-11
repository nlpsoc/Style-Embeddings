from unittest import TestCase
from STEL_error_analysis import ErrorAnalysis
import pandas as pd
from set_for_global import set_logging, set_global_seed

set_logging()
set_global_seed()


class TestGenerateFile(TestCase):
    def setUp(self) -> None:
        topic_rand_path = "/run/user/1000/gvfs/sftp:host=gw2hpcs03/hpc/uu_cs_nlpsoc/" \
                          "02-awegmann/style_discovery/results/topic-rand/"
        topic_sub_path = "/run/user/1000/gvfs/sftp:host=gw2hpcs03/hpc/uu_cs_nlpsoc/" \
                         "02-awegmann/style_discovery/results/topic-sub/"
        topic_conv_path = "/run/user/1000/gvfs/sftp:host=gw2hpcs03/hpc/uu_cs_nlpsoc/" \
                          "02-awegmann/style_discovery/results/topic-conv/"
        base_path = "/run/user/1000/gvfs/sftp:host=gw2hpcs03/hpc/uu_cs_nlpsoc/" \
                    "02-awegmann/style_discovery/results/base/"

        self.rand_path_BERTc5 = topic_rand_path + "Bert/contrastive/margin-0.5/" \
                                                  "03_STEL_single-predictions_topic-rand-bert-base-cased-" \
                                                  "loss-contrastive-margin-0.5-evaluator-binary__1638202028.258763.tsv"
        self.sub_path_BERTc5 = topic_sub_path + \
                               "Bert/contrastive/margin-0.5/" \
                               "03_STEL_single-predictions_topic-sub-bert-base-cased-" \
                               "loss-contrastive-margin-0.5-evaluator-binary__1638215769.9940262.tsv"
        self.conv_path_BERTc5 = topic_conv_path + \
                                "Bert/contrastive/margin-0.5/" \
                                "03_STEL_single-predictions_topic-conv-bert-base-cased-" \
                                "loss-contrastive-margin-0.5-evaluator-binary__1638181226.0273278.tsv"

        self.rand_path_BERTt5 = topic_rand_path + "Bert/triplet/margin-0.5/" \
                                                  "03_STEL_single-predictions_topic-rand-bert-base-cased" \
                                                  "-loss-triplet-margin-0.5-evaluator-triplet__1638195113.4850972.tsv"
        self.sub_path_BERTt5 = topic_sub_path + \
                               "Bert/triplet/margin-0.5/" \
                               "03_STEL_single-predictions_topic-sub-bert-base-cased-" \
                               "loss-triplet-margin-0.5-evaluator-triplet__1638222644.996105.tsv"
        self.conv_path_BERTt5 = topic_conv_path + \
                                "Bert/triplet/margin-0.5/" \
                                "03_STEL_single-predictions_topic-conv-bert-base-cased-" \
                                "loss-triplet-margin-0.5-evaluator-triplet__1638188188.488348.tsv"

        self.rand_path_RoBERTac5 = topic_rand_path + \
                                   "RoBERTa/contrastive/margin-0.5/" \
                                   "03_STEL_single-predictions_topic-rand-roberta-base-" \
                                   "loss-contrastive-margin-0.5-evaluator-binary__1638209141.5635898.tsv"
        self.sub_path_RoBERTac5 = topic_sub_path + \
                                  "RoBERTa/contrastive/margin-0.5/" \
                                  "03_STEL_single-predictions_topic-sub-roberta-base-" \
                                  "loss-contrastive-margin-0.5-evaluator-binary__1638229420.592142.tsv"
        self.conv_path_RoBERTac5 = topic_conv_path + \
                                   "RoBERTa/contrastive/margin-0.5/" \
                                   "03_STEL_single-predictions_topic-conv-roberta-base-" \
                                   "loss-contrastive-margin-0.5-evaluator-binary__1638331234.3157911.tsv"

        self.rand_path_RoBERTat5 = topic_rand_path + \
                                   "RoBERTa/triplet/margin-0.5/" \
                                   "03_STEL_single-predictions_topic-rand-roberta-base-" \
                                   "loss-triplet-margin-0.5-evaluator-triplet__1638423613.8016965.tsv"
        self.sub_path_RoBERTat5 = topic_sub_path + \
                                  "RoBERTa/triplet/margin-0.5/" \
                                  "03_STEL_single-predictions_topic-sub-roberta-base-loss-" \
                                  "triplet-margin-0.5-evaluator-triplet__1638236219.5952828.tsv"
        self.conv_path_RoBERTat5 = topic_conv_path + \
                                   "RoBERTa/triplet/margin-0.5/" \
                                   "03_STEL_single-predictions_topic-conv-roberta-base-" \
                                   "loss-triplet-margin-0.5-evaluator-triplet__1638209126.709481.tsv"

        self.rand_path_bertc5 = topic_rand_path + \
                                "bert/contrastive/margin-0.5/" \
                                "03_STEL_single-predictions_topic-rand-bert-base-uncased" \
                                "-loss-contrastive-margin-0.5-evaluator-binary__1638181388.4722538.tsv"
        self.sub_path_bertc5 = topic_sub_path + \
                               "bert/contrastive/margin-0.5/" \
                               "03_STEL_single-predictions_topic-sub-bert-base-uncased-" \
                               "loss-contrastive-margin-0.5-evaluator-binary__1637925215.4294097.tsv"
        self.conv_path_bertc5 = topic_conv_path + \
                                "bert/contrastive/margin-0.5/" \
                                "03_STEL_single-predictions_topic-conv-bert-base-uncased-" \
                                "loss-contrastive-margin-0.5-evaluator-binary__1638195226.4745457.tsv"

        self.rand_path_bertt5 = topic_rand_path + \
                                "bert/triplet/margin-0.5/" \
                                "03_STEL_single-predictions_topic-rand-bert-base-uncased" \
                                "-loss-triplet-margin-0.5-evaluator-triplet__1638188286.3812978.tsv"
        self.sub_path_bertt5 = topic_sub_path + \
                               "bert/triplet/margin-0.5/" \
                               "03_STEL_single-predictions_topic-sub-bert-base-uncased-" \
                               "loss-triplet-margin-0.5-evaluator-triplet__1638201976.2487254.tsv"
        self.conv_path_bertt5 = topic_conv_path + \
                                "bert/triplet/margin-0.5/" \
                                "03_STEL_single-predictions_topic-conv-bert-base-uncased-" \
                                "loss-triplet-margin-0.5-evaluator-triplet__1638202146.331307.tsv"

        self.path_bert = base_path + "bert/03_STEL_single-predictions_-bert-base-uncased__1638201763.8511195.tsv"
        self.path_BERT = base_path + "Bert/03_STEL_single-predictions_-bert-base-cased__1638208358.2290041.tsv"
        self.path_RoBERTa = base_path + "RoBERTa/03_STEL_single-predictions_-roberta-base__1638214904.9582798.tsv"

        self.sub_tuned = [self.sub_path_bertc5, self.sub_path_bertt5,
                          self.sub_path_BERTt5, self.sub_path_BERTc5,
                          self.sub_path_RoBERTac5, self.sub_path_RoBERTat5]
        self.rand_tuned = [self.rand_path_bertc5, self.rand_path_bertt5,
                           self.rand_path_BERTt5, self.rand_path_BERTc5,
                           self.rand_path_RoBERTac5, self.rand_path_RoBERTat5]
        self.conv_tuned = [self.conv_path_bertc5, self.conv_path_bertt5,
                           self.conv_path_BERTt5, self.conv_path_BERTc5,
                           self.conv_path_RoBERTac5, self.conv_path_RoBERTat5]
        self.base_paths = [self.path_bert, self.path_BERT, self.path_RoBERTa]

    def test_find_sub_unlearned(self):
        err_ann = ErrorAnalysis(sub_tuned=self.sub_tuned, rand_tuned=self.rand_tuned, conv_tuned=self.conv_tuned,
                                base_paths=self.base_paths)
        err_ann.find_un_learned_topic(topic="sub")

    def test_find_conv_unlearned(self):
        err_ann = ErrorAnalysis(sub_tuned=self.sub_tuned, rand_tuned=self.rand_tuned, conv_tuned=self.conv_tuned,
                                base_paths=self.base_paths)
        err_ann.find_un_learned_topic(topic="conv")

    def test_find_rand_unlearned(self):
        err_ann = ErrorAnalysis(sub_tuned=self.sub_tuned, rand_tuned=self.rand_tuned, conv_tuned=self.conv_tuned,
                                base_paths=self.base_paths)
        err_ann.find_un_learned_topic(topic="rand")

    def test_find_unlearned_model(self):
        err_ann = ErrorAnalysis(sub_tuned=self.sub_tuned, rand_tuned=self.rand_tuned, conv_tuned=self.conv_tuned,
                                base_paths=self.base_paths)
        err_ann.find_unlearned_model(model="bert")

    def test_find_unlearned_Bert(self):
        err_ann = ErrorAnalysis(sub_tuned=self.sub_tuned, rand_tuned=self.rand_tuned, conv_tuned=self.conv_tuned,
                                base_paths=self.base_paths)
        err_ann.find_unlearned_model(model="Bert")

    def test_find_unlearned_RoBerta(self):
        err_ann = ErrorAnalysis(sub_tuned=self.sub_tuned, rand_tuned=self.rand_tuned, conv_tuned=self.conv_tuned,
                                base_paths=self.base_paths)
        err_ann.find_unlearned_model(model="RoBERTa")

    def test_find_unlearned_all(self):
        err_ann = ErrorAnalysis(sub_tuned=self.sub_tuned, rand_tuned=self.rand_tuned, conv_tuned=self.conv_tuned,
                                base_paths=self.base_paths)
        err_ann.find_unlearned_all()

    def test_find_unlearned_single_rob_conv(self):
        err_ann = ErrorAnalysis(sub_tuned=self.sub_tuned, rand_tuned=self.rand_tuned, conv_tuned=self.conv_tuned,
                                base_paths=self.base_paths)
        err_ann.find_unlearned_stel_instances(base_paths=[self.path_RoBERTa],
                                              tuned_paths=[self.conv_path_RoBERTac5, self.conv_path_RoBERTat5])  #
        err_ann.find_learned_stel_instances(base_paths=[self.path_RoBERTa],
                                            tuned_paths=[self.conv_path_RoBERTac5, self.conv_path_RoBERTat5])  #

    def test_find_unlearned_single_rob_sub(self):
        err_ann = ErrorAnalysis(sub_tuned=self.sub_tuned, rand_tuned=self.rand_tuned, conv_tuned=self.conv_tuned,
                                base_paths=self.base_paths)
        err_ann.find_unlearned_stel_instances(base_paths=[self.path_RoBERTa],
                                              tuned_paths=[self.sub_path_RoBERTac5, self.sub_path_RoBERTat5])
        err_ann.find_learned_stel_instances(base_paths=[self.path_RoBERTa],
                                            tuned_paths=[self.sub_path_RoBERTac5, self.sub_path_RoBERTat5])

    def test_find_unlearned_single_rob_rand(self):
        err_ann = ErrorAnalysis(sub_tuned=self.sub_tuned, rand_tuned=self.rand_tuned, conv_tuned=self.conv_tuned,
                                base_paths=self.base_paths)
        err_ann.find_unlearned_stel_instances(base_paths=[self.path_RoBERTa],
                                              tuned_paths=[self.rand_path_RoBERTac5, self.rand_path_RoBERTat5])
        err_ann.find_learned_stel_instances(base_paths=[self.path_RoBERTa],
                                            tuned_paths=[self.rand_path_RoBERTac5, self.rand_path_RoBERTat5])

    def test_find_unlearned_stel_instances(self):
        err_ann = ErrorAnalysis()
        self.assertEqual(len(err_ann.stel), 1830)
        unlearned_stel_instances = err_ann.find_unlearned_stel_instances(
            base_paths=[self.path_bert, self.path_BERT, self.path_RoBERTa],
            tuned_paths=[self.rand_path_bertc5, self.rand_path_bertt5,
                         self.rand_path_BERTt5, self.rand_path_BERTc5,
                         self.rand_path_RoBERTac5, self.rand_path_RoBERTat5])

        BERT_results = pd.read_csv(self.path_BERT, sep='\t')
        for row_id, row in BERT_results[BERT_results['ID'].isin(unlearned_stel_instances['ID'])].iterrows():
            self.assertEqual(row['Correct Alternative'], row['TunedSentenceBertSimilarity'])
        bert_results = pd.read_csv(self.path_bert, sep='\t')
        for row_id, row in bert_results[bert_results['ID'].isin(unlearned_stel_instances['ID'])].iterrows():
            self.assertEqual(row['Correct Alternative'], row['TunedSentenceBertSimilarity'])
        RoBERTa_results = pd.read_csv(self.path_RoBERTa, sep='\t')
        for row_id, row in RoBERTa_results[RoBERTa_results['ID'].isin(unlearned_stel_instances['ID'])].iterrows():
            self.assertEqual(row['Correct Alternative'], row['TunedSentenceBertSimilarity'])

        print(unlearned_stel_instances)
        err_ann.print_sample_instances(unlearned_stel_instances)

        # unlearned_sample = unlearned_stel_instances[unlearned_stel_instances['style type'] == 'simplicity'].sample()
        # print(f"A1 - {unlearned_sample.iloc[0]['Anchor 1']}")
        # print(f"A2 - {unlearned_sample.iloc[0]['Anchor 2']}")
        # print(f"S1 - {unlearned_sample.iloc[0]['Alternative 1.1']}")
        # print(f"S2 - {unlearned_sample.iloc[0]['Alternative 1.2']}")
        # print(f"Correct Answer - {unlearned_sample.iloc[0]['Correct Alternative'] - 1}")
        # unlearned_sample = unlearned_stel_instances[unlearned_stel_instances['style type'] == 'simplicity'].sample()
        # print(f"A1 - {unlearned_sample.iloc[0]['Anchor 1']}")
        # print(f"A2 - {unlearned_sample.iloc[0]['Anchor 2']}")
        # print(f"S1 - {unlearned_sample.iloc[0]['Alternative 1.1']}")
        # print(f"S2 - {unlearned_sample.iloc[0]['Alternative 1.2']}")
        # print(f"Correct Answer - {unlearned_sample.iloc[0]['Correct Alternative'] - 1}")
        # unlearned_sample

        # pass


class SeedRoBERTa(TestCase):
    def setUp(self) -> None:
        topic_rand_path = "/run/user/1000/gvfs/sftp:host=gw2hpcs03/hpc/uu_cs_nlpsoc/" \
                          "02-awegmann/style_discovery/results/topic-rand/"
        topic_sub_path = "/run/user/1000/gvfs/sftp:host=gw2hpcs03/hpc/uu_cs_nlpsoc/" \
                         "02-awegmann/style_discovery/results/topic-sub/"
        topic_conv_path = "/run/user/1000/gvfs/sftp:host=gw2hpcs03/hpc/uu_cs_nlpsoc/" \
                          "02-awegmann/style_discovery/results/topic-conv/"
        base_path = "/run/user/1000/gvfs/sftp:host=gw2hpcs03/hpc/uu_cs_nlpsoc/" \
                    "02-awegmann/style_discovery/results/base/"

        self.rand_path_RoBERTac103 = topic_rand_path + "RoBERTa/contrastive/margin-0.5/seed-103/" \
                                                       "03_STEL_single-predictions_roberta-base-loss-contrastive-" \
                                                       "margin-0.5-evaluator-binary-seed-103__1640240513.9529076.tsv"
        self.rand_path_RoBERTac104 = topic_rand_path + "RoBERTa/contrastive/margin-0.5/seed-104/" \
                                                       "03_STEL_single-predictions_roberta-base-loss-contrastive-" \
                                                       "margin-0.5-evaluator-binary-seed-104__1640317852.0532126.tsv"
        self.rand_path_RoBERTac105 = topic_rand_path + "RoBERTa/contrastive/margin-0.5/seed-105/" \
                                                       "03_STEL_single-predictions_roberta-base-loss-contrastive-" \
                                                       "margin-0.5-evaluator-binary-seed-105__1640391793.4591277.tsv"

        self.rand_path_RoBERTat103 = topic_rand_path + "RoBERTa/triplet/margin-0.5/seed-103/" \
                                                       "03_STEL_single-predictions_roberta-base-loss-triplet-" \
                                                       "margin-0.5-evaluator-triplet-seed-103__1640447585.2321434.tsv"
        self.rand_path_RoBERTat104 = topic_rand_path + "RoBERTa/triplet/margin-0.5/seed-104/" \
                                                       "03_STEL_single-predictions_roberta-base-loss-triplet-" \
                                                       "margin-0.5-evaluator-triplet-seed-104__1640503629.3553493.tsv"
        self.rand_path_RoBERTat105 = topic_rand_path + "RoBERTa/triplet/margin-0.5/seed-105/" \
                                                       "03_STEL_single-predictions_roberta-base-loss-triplet-" \
                                                       "margin-0.5-evaluator-triplet-seed-105__1640559624.4107165.tsv"

        self.sub_path_RoBERTac104 = topic_sub_path + "RoBERTa/contrastive/margin-0.5/seed-104/" \
                                                     "03_STEL_single-predictions_roberta-base-loss-contrastive-" \
                                                     "margin-0.5-evaluator-binary-seed-104__1640307703.2167966.tsv"
        self.sub_path_RoBERTac105 = topic_sub_path + "RoBERTa/contrastive/margin-0.5/seed-105/" \
                                                     "03_STEL_single-predictions_roberta-base-loss-contrastive-" \
                                                     "margin-0.5-evaluator-binary-seed-105__1640380907.5737302.tsv"
        self.sub_path_RoBERTac106 = topic_sub_path + "RoBERTa/contrastive/margin-0.5/seed-106/" \
                                                     "03_STEL_single-predictions_roberta-base-loss-contrastive-" \
                                                     "margin-0.5-evaluator-binary-seed-106__1640744722.13345.tsv"

        self.sub_path_RoBERTat104 = topic_sub_path + "RoBERTa/triplet/margin-0.5/seed-104/" \
                                                     "03_STEL_single-predictions_roberta-base-loss-triplet-" \
                                                     "margin-0.5-evaluator-triplet-seed-104__1640493002.02451.tsv"
        self.sub_path_RoBERTat105 = topic_sub_path + "RoBERTa/triplet/margin-0.5/seed-105/" \
                                                     "03_STEL_single-predictions_roberta-base-loss-triplet-" \
                                                     "margin-0.5-evaluator-triplet-seed-105__1640549560.3098667.tsv"
        self.sub_path_RoBERTat106 = topic_sub_path + "RoBERTa/triplet/margin-0.5/seed-106/" \
                                                     "03_STEL_single-predictions_roberta-base-loss-triplet-" \
                                                     "margin-0.5-evaluator-triplet-seed-106__1640801227.764549.tsv"

        self.conv_path_RoBERTac103 = topic_conv_path + "RoBERTa/contrastive/margin-0.5/seed-103/" \
                                                       "03_STEL_single-predictions_roberta-base-loss-contrastive-" \
                                                       "margin-0.5-evaluator-binary-seed-103__1640234905.3592918.tsv"
        self.conv_path_RoBERTac107 = topic_conv_path + "RoBERTa/contrastive/margin-0.5/seed-107/" \
                                                       "03_STEL_single-predictions_roberta-base-loss-contrastive-" \
                                                       "margin-0.5-evaluator-binary-seed-107__1640818761.8924484.tsv"
        self.conv_path_RoBERTac108 = topic_conv_path + "RoBERTa/contrastive/margin-0.5/seed-108/" \
                                                       "03_STEL_single-predictions_roberta-base-loss-contrastive-" \
                                                       "margin-0.5-evaluator-binary-seed-108__1641259016.1090963.tsv"

        self.conv_path_RoBERTat104 = topic_conv_path + "RoBERTa/triplet/margin-0.5/seed-104/" \
                                                       "03_STEL_single-predictions_roberta-base-loss-triplet-" \
                                                       "margin-0.5-evaluator-triplet-seed-104__1640495848.0824268.tsv"
        self.conv_path_RoBERTat105 = topic_conv_path + "RoBERTa/triplet/margin-0.5/seed-105/" \
                                                       "03_STEL_single-predictions_roberta-base-loss-triplet-" \
                                                       "margin-0.5-evaluator-triplet-seed-105__1640552965.3741663.tsv"
        self.conv_path_RoBERTat106 = topic_conv_path + "RoBERTa/triplet/margin-0.5/seed-106/" \
                                                       "03_STEL_single-predictions_roberta-base-loss-triplet-" \
                                                       "margin-0.5-evaluator-triplet-seed-106__1640874528.8494937.tsv"

        self.path_RoBERTa = base_path + "RoBERTa/03_STEL_single-predictions_-roberta-base__1638214904.9582798.tsv"

        self.sub_tuned = [self.sub_path_RoBERTac104, self.sub_path_RoBERTac105, self.sub_path_RoBERTac106,
                          self.sub_path_RoBERTat104, self.sub_path_RoBERTat105, self.sub_path_RoBERTat106]
        self.rand_tuned = [self.rand_path_RoBERTac103, self.rand_path_RoBERTac104, self.rand_path_RoBERTac105,
                           self.rand_path_RoBERTat103, self.rand_path_RoBERTat104, self.rand_path_RoBERTat105]
        self.conv_tuned = [self.conv_path_RoBERTac103, self.conv_path_RoBERTac107, self.conv_path_RoBERTac108,
                           self.conv_path_RoBERTat104, self.conv_path_RoBERTat105, self.conv_path_RoBERTat106]
        self.trip_tuned = [self.sub_path_RoBERTat104, self.sub_path_RoBERTat105, self.sub_path_RoBERTat106,
                           self.rand_path_RoBERTat103, self.rand_path_RoBERTat104, self.rand_path_RoBERTat105,
                           self.conv_path_RoBERTat104, self.conv_path_RoBERTat105, self.conv_path_RoBERTat106]
        self.con_tuned = [self.sub_path_RoBERTac104, self.sub_path_RoBERTac105, self.sub_path_RoBERTac106,
                          self.rand_path_RoBERTac103, self.rand_path_RoBERTac104, self.rand_path_RoBERTac105,
                          self.conv_path_RoBERTac103, self.conv_path_RoBERTac107, self.conv_path_RoBERTac108]
        self.base_paths = [self.path_RoBERTa]


    def test_find_unlearned_all(self):
        # all R in Table 5
        err_ann = ErrorAnalysis(sub_tuned=self.sub_tuned, rand_tuned=self.rand_tuned, conv_tuned=self.conv_tuned,
                                base_paths=self.base_paths)
        err_ann.find_unlearned_all()


    def test_find_sub_unlearned(self):
        # s R in Table 5
        err_ann = ErrorAnalysis(sub_tuned=self.sub_tuned, rand_tuned=self.rand_tuned, conv_tuned=self.conv_tuned,
                                base_paths=self.base_paths)
        err_ann.find_un_learned_topic(topic="sub")

    def test_find_conv_unlearned(self):
        # c R in Table 5
        err_ann = ErrorAnalysis(sub_tuned=self.sub_tuned, rand_tuned=self.rand_tuned, conv_tuned=self.conv_tuned,
                                base_paths=self.base_paths)
        err_ann.find_un_learned_topic(topic="conv")

    def test_find_rand_unlearned(self):
        # r R in Table 5
        err_ann = ErrorAnalysis(sub_tuned=self.sub_tuned, rand_tuned=self.rand_tuned, conv_tuned=self.conv_tuned,
                                base_paths=self.base_paths)
        err_ann.find_un_learned_topic(topic="rand")

    def test_find_t_unlearned(self):
        err_ann = ErrorAnalysis(trip_tuned=self.trip_tuned, base_paths=self.base_paths)
        err_ann.find_un_learned_loss(loss="trip")

    def test_find_c_unlearned(self):
        err_ann = ErrorAnalysis(con_tuned=self.con_tuned, base_paths=self.base_paths)
        err_ann.find_un_learned_loss(loss="con")

    def test_find_union_unlearned(self):
        error_analysis = ErrorAnalysis(con_tuned=self.con_tuned, base_paths=self.base_paths, trip_tuned=self.trip_tuned,
                                       sub_tuned=self.sub_tuned, rand_tuned=self.rand_tuned, conv_tuned=self.conv_tuned)
