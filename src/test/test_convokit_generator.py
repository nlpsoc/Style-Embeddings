from unittest import TestCase
from convokit_generator import TaskGenerator, ConvokitStream
import generate_dataset
from global_const import SUBREDDIT_U2_COL, SUBREDDIT_U1_COL, SUBREDDIT_A_COL, CONVERSATION_U2_COL, CONVERSATION_U1_COL, \
    CONVERSATION_A_COL, ID_U2_COL, ID_U1_COL, ID_A_COL, AUTHOR_U2_COL, AUTHOR_U1_COL, AUTHOR_A_COL, TOPIC_SUBREDDIT, \
    TOPIC_RANDOM, TOPIC_CONVERSATION, set_logging
import pandas
from pathlib import Path
import logging

set_logging()


class TestTrainTripleGenerator(TestCase):
    def setUp(self) -> None:
        self.set_to_years = [2012, 2018]
        self.set_to_subreddits = ["subreddit-Cornell", "subreddit-ApplyingToCollege"]
        self.conv_dir = str(Path.home()) + "/convo-test"

    def test_get_data_split_unknown_var(self):
        logging.info(self.conv_dir)
        triple_gen = TaskGenerator(convokit_data_keys=[self.set_to_subreddits[0]], years=[self.set_to_years[0]],
                                   total=10)
        self.assertRaises(AssertionError, triple_gen._get_data_split, None, None, None, 'dummy')

    def test_get_data_split_same_conv(self):
        triple_gen = TaskGenerator(convokit_data_keys=[self.set_to_subreddits[0]], years=[self.set_to_years[0]],
                                   total=10)
        train_data, dev_data, test_data = triple_gen._get_data_split(topic_variable=TOPIC_CONVERSATION)
        train_authors = train_data[[AUTHOR_A_COL, AUTHOR_U1_COL, AUTHOR_U2_COL]].values.ravel()
        train_authors = pandas.unique(train_authors)
        dev_authors = dev_data[[AUTHOR_A_COL, AUTHOR_U1_COL, AUTHOR_U2_COL]].values.ravel()
        dev_authors = pandas.unique(dev_authors)
        test_authors = test_data[[AUTHOR_A_COL, AUTHOR_U1_COL, AUTHOR_U2_COL]].values.ravel()
        test_authors = pandas.unique(test_authors)
        # test that the authors are disjoint
        self.assertTrue(set(train_authors).isdisjoint(set(dev_authors)))
        self.assertTrue(set(train_authors).isdisjoint(set(test_authors)))
        self.assertTrue(set(dev_authors).isdisjoint(set(test_authors)))
        self.assertTrue(all((values[CONVERSATION_A_COL] == values[CONVERSATION_U1_COL]) or
                            (values[CONVERSATION_A_COL] == values[CONVERSATION_U2_COL])
                            for _, values in train_data.iterrows()))

        print(train_data)
        print(dev_data)
        print(test_data)

    def test_get_data_split_random(self):
        triple_gen = TaskGenerator(convokit_data_keys=[self.set_to_subreddits[0]], years=[self.set_to_years[0]],
                                   total=100)
        train_data, dev_data, test_data = triple_gen._get_data_split(topic_variable=TOPIC_RANDOM)
        train_authors = train_data[[AUTHOR_A_COL, AUTHOR_U1_COL, AUTHOR_U2_COL]].values.ravel()
        train_authors = pandas.unique(train_authors)
        dev_authors = dev_data[[AUTHOR_A_COL, AUTHOR_U1_COL, AUTHOR_U2_COL]].values.ravel()
        dev_authors = pandas.unique(dev_authors)
        test_authors = test_data[[AUTHOR_A_COL, AUTHOR_U1_COL, AUTHOR_U2_COL]].values.ravel()
        test_authors = pandas.unique(test_authors)
        # test that the authors are disjoint
        self.assertTrue(set(train_authors).isdisjoint(set(dev_authors)))
        self.assertTrue(set(train_authors).isdisjoint(set(test_authors)))
        self.assertTrue(set(dev_authors).isdisjoint(set(test_authors)))
        self.assertFalse(all((row[CONVERSATION_A_COL] == row[CONVERSATION_U1_COL]) or
                             (row[CONVERSATION_A_COL] == row[CONVERSATION_U2_COL])
                             for _, row in train_data.iterrows()))

        print(train_data)
        print(dev_data)
        print(test_data)

    def test_get_data_split_subreddit(self):
        triple_gen = TaskGenerator(convokit_data_keys=self.set_to_subreddits, years=[self.set_to_years[0]],
                                   total=10, directory=self.conv_dir)
        train_data, dev_data, test_data = triple_gen._get_data_split(topic_variable=TOPIC_SUBREDDIT)
        train_authors = train_data[[AUTHOR_A_COL, AUTHOR_U1_COL, AUTHOR_U2_COL]].values.ravel()
        train_authors = pandas.unique(train_authors)
        dev_authors = dev_data[[AUTHOR_A_COL, AUTHOR_U1_COL, AUTHOR_U2_COL]].values.ravel()
        dev_authors = pandas.unique(dev_authors)
        test_authors = test_data[[AUTHOR_A_COL, AUTHOR_U1_COL, AUTHOR_U2_COL]].values.ravel()
        test_authors = pandas.unique(test_authors)
        # test that the authors are disjoint
        self.assertTrue(set(train_authors).isdisjoint(set(dev_authors)))
        self.assertTrue(set(train_authors).isdisjoint(set(test_authors)))
        self.assertTrue(set(dev_authors).isdisjoint(set(test_authors)))
        # TEST if all tasks fulfill that (A, DA) are in the same subreddit
        self.assertTrue(all((values[SUBREDDIT_A_COL] == values[SUBREDDIT_U1_COL] and
                             values[AUTHOR_A_COL] != values[AUTHOR_U1_COL]) or
                            (values[SUBREDDIT_A_COL] == values[SUBREDDIT_U2_COL] and
                             values[AUTHOR_A_COL] != values[AUTHOR_U2_COL])
                            for _, values in train_data.iterrows()))
        # TEST if there exist tasks where (A, U1) and (A, U2) BOTH do not occur in the same conversation
        self.assertFalse(all((values[CONVERSATION_A_COL] == values[CONVERSATION_U1_COL]) or
                             (values[CONVERSATION_A_COL] == values[CONVERSATION_U2_COL])
                             for _, values in train_data.iterrows()))

        print(train_data)
        print(dev_data)
        print(test_data)

        print(train_data[[CONVERSATION_A_COL, CONVERSATION_U1_COL, CONVERSATION_U2_COL]])
        print(train_data[[SUBREDDIT_A_COL, SUBREDDIT_U1_COL, SUBREDDIT_U2_COL]])

    def test_main(self):
        result_dict = generate_dataset.main(total=10, convokit_keys=["subreddit-Cornell",
                                                                     "subreddit-ApplyingToCollege"],
                                            convo_directory=self.conv_dir,
                                            debug=True, convs_per_subreddit=10)

        conv_train = result_dict["conversation"][0]
        sub_train = result_dict["subreddit"][0]
        rand_train = result_dict["random"][0]

        # TEST: (A, SA) is the same for all three
        for i, task_df in conv_train.iterrows():
            # Is SA U1 or U2?
            sa = ID_U1_COL
            if task_df[AUTHOR_A_COL] == task_df[AUTHOR_U2_COL]:
                sa = ID_U2_COL
            self.assertEqual(sub_train.iloc[i][sa], task_df[sa])
            self.assertEqual(sub_train.iloc[i][ID_A_COL], task_df[ID_A_COL])
            self.assertEqual(rand_train.iloc[i][sa], task_df[sa])
            self.assertEqual(rand_train.iloc[i][ID_A_COL], task_df[ID_A_COL])

    # def test_w_mergedsample(self):
    #     big_sample = "sample-reddit-2018"
    #     convo_stream = ConvokitStream(convokit_data_keys=[big_sample], directory=self.conv_dir,
    #                                   convo_per_sub=None, years=None, min_comments_convo=None)
    #     self.assertEqual(600*100, len(convo_stream.corpus.get_conversation_ids()))
    # def test_smallmerged(self):
    #     small_sample = "merged-sample"
    #     output_dir = '/home/anna/Documents/UU/digital-society_intergenerational-empathy/src/output'
    #     train_f = 'train-7__subreddits-2_year-2018-2018_tasks-10_topic_variable-conversation.tsv'
    #     dev_f = 'dev-1__subreddits-2_year-2018-2018_tasks-10_topic_variable-conversation.tsv'
    #     test_f = 'test-1__subreddits-2_year-2018-2018_tasks-10_topic_variable-conversation.tsv'
    #     task_gen = TaskGenerator(convokit_data_keys=[small_sample], directory=self.conv_dir, convo_per_sub=None,
    #                              author_data_f=output_dir + "/author_data.json", total=10)
    #     task_gen.load_authors_set_from_json(output_dir)
    #     task_gen.get_data_split(author_data_f=output_dir + "/author_data.json", train_data_f=output_dir + "/" + train_f,
    #                             dev_data_f=output_dir + "/" + dev_f, test_data_f=output_dir + "/" + test_f,
    #                             topic_variable=TOPIC_SUBREDDIT)
    # def test_main_merged(self):
    #     generate_dataset.main(convokit_keys=['merged-sample'],
    #                           total=1000, convo_directory=self.conv_dir, debug=True)
    #
    # def test_main_mergedsample(self):
    #     big_sample = "sample-reddit-2018"
    #
    #     generate_dataset.main(convokit_keys=[big_sample],
    #                           total=1000, convo_directory=self.conv_dir, debug=True)


class TestConvokitStream(TestCase):
    def setUp(self) -> None:
        self.set_to_years = [2012, 2018]
        # self.set_to_subreddits = ["subreddit-Cornell", "subreddit-ApplyingToCollege"]
        self.set_to_subreddits = ["subreddit-Cornell", "subreddit-BostonTerrier"]
        self.conv_dir = "/home/anna/convo-test"
        self.convo_per_sub = 100
        self.min_comments_per_conv = 10

    def test_single_stream(self):
        convo_stream = ConvokitStream([self.set_to_subreddits[0]], years=[self.set_to_years[1]],
                                      min_comments_convo=self.min_comments_per_conv, directory=self.conv_dir,
                                      convo_per_sub=self.convo_per_sub)  # ["subreddit-Cornell", "subreddit-ApplyingToCollege"]
        num_convs = 0
        rand_convos = ['95maej', '98whof', '7zmvva', '8dtupn', '8g4b4p', '9bz7gy', '9bdat5', '97z0ye', '8ag17u',
                       '9bxmg0', '84jbwt', '8uy706', '8hbxm8', '810tnb', '93ma7q', '9l1oy2', '8f0nok', '84yw94',
                       '905jyt', '8iy4bh', '7untrm', '9dzvbh', '820z4t', '8bvkg1', '8yo86l', '9lpkya', '8l3xht',
                       '9m1fy5', '7v4scp', '7qdfms', '875r6v', '7x21wl', '9n7m02', '98k1fe', '97yxnt', '97kw07',
                       '8rugvr', '9d97yd', '86l3ne', '8ja58j', '9hwd5c', '8y2mj7', '8jps82', '8x5pu6', '7yy032',
                       '89dfrz', '9qi4wi', '9fvwkm', '9gvd8h', '8qf1u8', '8g0vlu', '8zh6j9', '7woise', '8cz7li',
                       '88e1sq', '8ur7ys', '8oanmh', '9ogt74', '8vjnc7', '90z6jm', '8gsbbr', '8wb49w', '9qtai8',
                       '8znvd4', '96ywv8', '9c5rew', '8n8t3h', '84jhm4', '9jurch', '91n05s', '8hpku1', '9sv4dm',
                       '88aigq', '8rh6cf', '9oz71e', '9mqgpg', '8sjzmk', '8m3ts1', '9nll3o', '9o4syu', '7ody3s',
                       '9j92k7', '82b7hj', '8t69jn', '8neem0', '9ockj3', '9iyo8u', '83qcao', '95q1u1', '8q9g7r',
                       '9mve8x', '8dggog', '8eg5gl', '9nl8rp', '844tqu', '83wo48', '8jeoeg', '94pxh3', '84gw96',
                       '8sf109']

        for i, conv in enumerate(convo_stream):
            year = ConvokitStream.get_year_from_timestamp(conv.meta["timestamp"])
            self.assertIn(year, [self.set_to_years[1]])
            self.assertGreaterEqual(conv.meta["num_comments"], self.min_comments_per_conv)
            num_convs += 1
            self.assertIn(conv.id, rand_convos)

        self.assertEqual(num_convs, self.convo_per_sub)

    def test_merge_stream(self):
        # From Documentation:
        # Utterances with the same id must share the same data,
        #   otherwise the other corpus utterance data & metadata will be ignored.
        #   A warning is printed when this happens.
        convo_stream = ConvokitStream(self.set_to_subreddits, years=[self.set_to_years[1]],
                                      min_comments_convo=10,
                                      convo_per_sub=100, directory=self.conv_dir)
        # ["subreddit-Cornell", "subreddit-ApplyingToCollege"]
        sub1_corpus = ConvokitStream([self.set_to_subreddits[0]], years=[self.set_to_years[1]], min_comments_convo=10,
                                     convo_per_sub=100, directory=self.conv_dir).get_corpus()
        sub2_corpus = ConvokitStream([self.set_to_subreddits[1]], years=[self.set_to_years[1]], min_comments_convo=10,
                                     convo_per_sub=100, directory=self.conv_dir).get_corpus()

        seen_subreddits = set()
        for i, conv in enumerate(convo_stream):
            # if i < 100:
            conv_id = conv.id
            if sub2_corpus.has_conversation(conv_id):
                self.assertEqual(sub2_corpus.get_conversation(conv_id).meta["subreddit"], conv.meta["subreddit"])
            if sub1_corpus.has_conversation(conv_id):
                self.assertEqual(sub1_corpus.get_conversation(conv_id).meta["subreddit"], conv.meta["subreddit"])
            seen_subreddits.add(conv.meta["subreddit"])
            # if len(seen_subreddits) == 2:
            #     break
            year = ConvokitStream.get_year_from_timestamp(conv.meta["timestamp"])
            self.assertIn(year, self.set_to_years)

        merged_corpus = convo_stream.get_corpus()
        # assert merged corpus contains the same numebr of utterances, conversations and speakers
        self.assertTrue(set(sub1_corpus.speakers).issubset(set(merged_corpus.speakers)))
        self.assertTrue(set(sub2_corpus.speakers).issubset(set(merged_corpus.speakers)))
        self.assertEqual(len(sub1_corpus.utterances) + len(sub2_corpus.utterances), len(merged_corpus.utterances))
        self.assertEqual(len(sub1_corpus.conversations) + len(sub2_corpus.conversations),
                         len(merged_corpus.conversations))
        self.assertEqual(len(seen_subreddits), 2)

    def test_manually_extract_year(self):
        convo_stream = ConvokitStream(["subreddit-Bowyer"], years=[self.set_to_years[1]],
                                      min_comments_convo=self.min_comments_per_conv, directory=self.conv_dir,
                                      convo_per_sub=self.convo_per_sub, manually_delete=True)

        num_convs = 0
        for i, conv in enumerate(convo_stream):
            year = ConvokitStream.get_year_from_timestamp(conv.meta["timestamp"])
            self.assertIn(year, [self.set_to_years[1]])
            self.assertGreaterEqual(conv.meta["num_comments"], self.min_comments_per_conv)
            num_convs += 1

        self.assertEqual(num_convs, self.convo_per_sub)
