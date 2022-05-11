"""
    Iterator over convokit Reddit data to use for training data generation
"""

# from smart_open import open
import logging
import random
import math
import json
import os

import pandas as pd
from convokit import Corpus, download, download_local
import sklearn
from typing import List

# GENERAL:
#   1 stands for distinct (or style change)
#   0 stands for same (or no style change)

# CONSTANTS for Reddit
from global_identifiable import CONVO_CACHE
from global_const import SUB_LIST, SAMPLE_YEARS, MIN_VALID_UTTS, CONVS_PER_SUB, MIN_COM_PER_CONV, SAME_AUTHOR_AU1_COL, \
    SUBREDDIT_U2_COL, SUBREDDIT_U1_COL, SUBREDDIT_A_COL, CONVERSATION_U2_COL, CONVERSATION_U1_COL, CONVERSATION_A_COL, \
    ID_U2_COL, ID_U1_COL, ID_A_COL, AUTHOR_U2_COL, AUTHOR_U1_COL, AUTHOR_A_COL, U2_COL, U1_COL, ANCHOR_COL, \
    TOPIC_SUBREDDIT, TOPIC_RANDOM, TOPIC_CONVERSATION, set_global_seed

# taken from STEL
REDDIT_SKIP_COMMENTS = [" ", "", " [removed] ", "[ removed ]", "[removed]", "[ deleted ]", "[deleted]", " [deleted] ",
                        "\n",
                        " \n", "\n ", " \n "]
TRAIN_RATIO = 0.7
TEST_RATIO = 0.15
DEV_RATIO = 0.15
MAX_TRY_UTT = 50  # do not try more than 50 times to select a valid utterance by an author in the same conv.
CONVOKIT_MERGE_KEYS = ["subreddit-IAmA", "subreddit-techsupport", "subreddit-hockey",
                       "subreddit-conspiracy", "subreddit-politics"]  #
# CONVOKIT_MERGE_KEYS = ["subreddit-ApplyingToCollege", "subreddit-Cornell"]
TOTAL = 500000

# CONSTANTS for politeness
WIKIPEDIA_POLITENESS = "wikipedia-politeness-corpus"


# ------------------------------- USING convokit reddit data -----------------------------------------------------

class ConvokitStream:
    """
        iterator for conversations of convokit_data_keys subreddits
    """
    MIN_NUM_COMMENTS = "min_num_comments"
    FILTERED = "filtered"

    def __init__(self, convokit_data_keys: List[str] = SUB_LIST, directory=CONVO_CACHE, years: List[int] = SAMPLE_YEARS,
                 d_only=False, min_comments_convo=MIN_COM_PER_CONV, convo_per_sub=CONVS_PER_SUB, manually_delete=False):
        """
            download subreddit data, delete conversations that are not in the years that interest us to save space,
            sample as many conversations as interested in per subreddit and merge them in a convokti corpus object

            :param convokit_data_keys: which subreddits to consider
            :param directory: convokit cache path
            :param years: which years to consider for the subreddits
            :param d_only: when this is called to only download the subreddits and not actually iterate over
                conversations, no merge of corpora is necessary which saves a lot of computation time
            :param min_comments_convo: only consider conversations with at least these many comments
            :param convo_per_sub: sample convo_per_sub number of conversations
            :param manually_delete: to reduce disk space delete lines from corpus that contain conversations of years
                that are not considered (i.e., not in years). Does nothing if years is empty
        """
        # self.text_cleaner = text_processing.textCleaner.TextCleaner()
        for i, convo_key in enumerate(convokit_data_keys):
            # Get Corpus
            try:
                downloaded_path = download_local(convo_key, data_dir=directory)
                logging.info("Found corpus {} locally at: {}...".format(convo_key, downloaded_path))
            except FileNotFoundError:
                logging.info("Downloading corpus {} ...".format(convo_key))
                downloaded_path = download(convo_key, data_dir=directory)

            if manually_delete and years:
                f_utt_jsonl = downloaded_path + '/utterances.jsonl'
                logging.info("Manually delete utterances from file {} for years {}".format(f_utt_jsonl, years))
                self.delete_lines(f_utt_jsonl, years)
                # f_conv_json = downloaded_path + '/conversations.json'
                # instead of altering conversations.json remove with filter after loading in ...

            logging.info("loading corpus from {}".format(downloaded_path))
            cur_corpus = Corpus(downloaded_path)
            # Remove information that will be incorrect after corpus merge
            if "subreddit" in cur_corpus.meta:
                subreddit = cur_corpus.meta["subreddit"]
                cur_corpus.meta["subreddits"] = [subreddit]
                self.delete_metadata(cur_corpus)

            corpus_altered = False
            # Filter for relevant year(s) if not done before
            if years is not None:
                # logging.info(cur_corpus.meta)
                if self.FILTERED not in cur_corpus.meta:
                    logging.info("Filtering conversations for relevant years ...")
                    cur_corpus = cur_corpus.filter_conversations_by(
                        lambda conv: ConvokitStream.get_year_from_timestamp(conv.meta["timestamp"]) in years)
                    # ATTENTION: replace with the smaller corpus on disk
                    logging.info("Replacing the local files with the filtered corpus at {}".format(downloaded_path))
                    self.update_meta_info(cur_corpus, years)
                    corpus_altered = True
                else:
                    logging.info("Filtering was already done in a previous step for years {}..."
                                 .format(cur_corpus.meta[self.FILTERED]))

            # Filter for minimal conversation length
            if min_comments_convo is not None:
                logging.info(cur_corpus.meta)
                if self.MIN_NUM_COMMENTS not in cur_corpus.meta:
                    logging.info("Filtering conversations with less than {} posts ...".format(min_comments_convo))
                    cur_corpus = cur_corpus.filter_conversations_by(
                        lambda conv: conv.meta["num_comments"] >= min_comments_convo
                    )
                    # ATTENTION: replace with the smaller corpus on disk
                    logging.info("Replacing the local files with the filtered corpus at {}".format(downloaded_path))
                    cur_corpus.add_meta(self.MIN_NUM_COMMENTS, min_comments_convo)
                    cur_corpus.meta.index.indices[self.MIN_NUM_COMMENTS] = ["<class 'int'>"]
                    cur_corpus.meta_index.overall_index[self.MIN_NUM_COMMENTS] = ["<class 'int'>"]
                    corpus_altered = True
                else:
                    logging.info("Filtering was already done in a previous step for maximum of {} comments..."
                                 .format(cur_corpus.meta[self.MIN_NUM_COMMENTS]))

            if corpus_altered:
                logging.info(cur_corpus.meta)
                cur_corpus.dump(name="", save_to_existing_path=True, fields_to_skip=None)

            # randomly select convo_per_sub conversations from subreddit
            if convo_per_sub:
                conv_ids = cur_corpus.get_conversation_ids()
                set_global_seed(w_torch=False)
                assert len(conv_ids) > convo_per_sub, "{} are note enough convos to extract {} from" \
                    .format(len(conv_ids), convo_per_sub)
                rand_convos = random.sample(conv_ids, convo_per_sub)
                cur_corpus = cur_corpus.filter_conversations_by(
                    lambda conv: conv.id in rand_convos
                )

            if d_only:
                del cur_corpus
                continue
            if i == 0:
                self.corpus = cur_corpus
            else:
                logging.info("Merging with previous corpora ...")
                self.corpus.meta["subreddits"].append(cur_corpus.meta["subreddits"][0])
                assert self.corpus.meta[self.MIN_NUM_COMMENTS] == cur_corpus.meta[self.MIN_NUM_COMMENTS]
                assert self.corpus.meta[self.FILTERED] == cur_corpus.meta[self.FILTERED], \
                    "values {} and {} do not match".format(self.corpus.meta[self.FILTERED],
                                                           cur_corpus.meta[self.FILTERED])
                del cur_corpus.meta[self.FILTERED]
                del cur_corpus.meta[self.MIN_NUM_COMMENTS]
                del cur_corpus.meta["subreddits"]
                self.corpus = self.corpus.merge(cur_corpus)
        if not d_only:
            self.corpus.print_summary_stats()
            # save the merged corpus
            self.corpus.dump(name="merged-sample", base_path=directory)

        # self.corpus = self.text_cleaner.transform(self.corpus)

    def update_meta_info(self, cur_corpus, years):
        cur_corpus.add_meta(self.FILTERED, years)
        # for some reason setting index and meta_index is necessary to be able to SAVE metadata
        cur_corpus.meta.index.indices[self.FILTERED] = ["<class 'list'>"]
        cur_corpus.meta_index.overall_index[self.FILTERED] = ["<class 'list'>"]

    @staticmethod
    def delete_lines(original_file, years):
        """ Delete a line from a file at the given line number """
        from datetime import datetime
        is_altered = False
        current_index = 0
        dummy_file = original_file + '.bak'
        # Open original file in read only mode and dummy file in write mode
        with open(original_file, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
            # Line by line copy data from original file to dummy file
            for line in read_obj:
                # If current line number matches the given line number then skip copying
                if int(datetime.utcfromtimestamp(json.loads(line)["timestamp"]).strftime("%Y")) in years:
                    write_obj.write(line)
                else:
                    is_altered = True
                current_index += 1
        # If any line is skipped then rename dummy file as original file
        if is_altered:
            os.remove(original_file)
            os.rename(dummy_file, original_file)
        else:
            os.remove(dummy_file)

    @staticmethod
    def delete_metadata(cur_corpus):
        if "subreddit" in cur_corpus.meta:
            del cur_corpus.meta["subreddit"]
        if "num_posts" in cur_corpus.meta:
            del cur_corpus.meta["num_posts"]
        if "num_comments" in cur_corpus.meta:
            del cur_corpus.meta["num_comments"]
        if "num_user" in cur_corpus.meta:
            del cur_corpus.meta["num_user"]

    @staticmethod
    def get_year_from_timestamp(timestamp: int) -> int:
        from datetime import datetime
        return int(datetime.utcfromtimestamp(timestamp).strftime('%Y'))

    def __iter__(self):
        """
            Iterate over all conversations in the self.corpus variable
        :return:
        """
        # for utt in self.corpus.iter_utterances():
        #     yield self.text_cleaner.transform_utterance(utt).text
        for conv in self.corpus.iter_conversations():
            yield conv

    def get_corpus(self):
        return self.corpus


class TaskGenerator:
    """
        class to generate CAV tasks with different content control variables
    """

    def __init__(self, convokit_data_keys: List[str] = SUB_LIST, years: List[int] = SAMPLE_YEARS, directory=CONVO_CACHE,
                 min_valid_utts=MIN_VALID_UTTS, total=TOTAL, print_stats=True, convo_per_sub=CONVS_PER_SUB,
                 author_data_f: str=None):
        """

        :param convokit_data_keys:
        :param subreddit:
        :param directory:
        :param min_valid_utts:  number of valid utterances that should be present per first author;
        this condition is only required approximately by requiring min_valid_utts + 0.1*min_valid_utts nonemtpy utterances
        per first_author; first authors that still contain too few valid utterances are removed during training
        :param upper_thresh:
        :param total: number of tasks to generate
        :param rel_test_set_size:
        """
        # if convokit_data_key is None:
        #    # TODO: memory error when assigning to corpus?
        #    self.corpus = Corpus(download("subreddit-" + subreddit, data_dir=directory))
        # else:
        self.convokit_keys = convokit_data_keys
        self.corpus = ConvokitStream(
            self.convokit_keys, years=years, convo_per_sub=convo_per_sub,
            directory=directory).get_corpus()  # Corpus(download(self.convokit_key, data_dir=directory))
        if print_stats:
            self.corpus.print_summary_stats()
        self.min_valid_utts = min_valid_utts
        # self.upper_thresh = upper_thresh
        self.nbr_triples_to_extract = total
        # self.test_size = rel_test_set_size
        self.train_ratio = TRAIN_RATIO
        self.dev_ratio = DEV_RATIO
        self.test_ratio = TEST_RATIO
        if not author_data_f:
            self._init_authors()  # set self.train_authors, self.train_superset, ...
        else:
            self.load_authors_set_from_json(os.path.dirname(author_data_f), os.path.basename(author_data_f))
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        # self.au2_sameconv = au2_sameconv
        # if self.au2_sameconv == False:
        #     logging.info('A and U2 are not required to come from the same conversation ...')

    def get_data_split(self, author_data_f: str, train_data_f: str, dev_data_f: str, test_data_f: str,
                       topic_variable: str = TOPIC_SUBREDDIT):
        self.load_authors_set_from_json(data_dir=os.path.dirname(author_data_f),
                                        filename=os.path.basename(author_data_f))
        train_data = pd.read_csv(train_data_f, sep='\t')
        dev_data = pd.read_csv(dev_data_f, sep='\t')
        test_data = pd.read_csv(test_data_f, sep='\t')
        self._get_data_split(train_data=train_data, dev_data=dev_data, test_data=test_data,
                             topic_variable=topic_variable)

    def _get_data_split(self, train_data=None, dev_data=None, test_data=None, topic_variable=TOPIC_CONVERSATION):
        """
        generates pandas dataframes for the 70%, 15%, 15% train, dev, test split

        :param topic_variable: has to be part of global_const.TOPIC_SUBREDDIT, global_const.TOPIC_RANDOM,
            global_const.TOPIC_CONVERSATION,
        :param train_data: pandas dataframe from which (A, U1) are supposed ot be taken
        :param dev_data: -"-
        :param test_data: -"-
        :return:
        """
        if train_data is not None or dev_data is not None or test_data is not None:
            logging.info('With the previously generated (A, SA) pairs generate ...')
            assert train_data is not None and dev_data is not None and test_data is not None  # and not ada_sameconv

        logging.info('For topic control level ={}'.format(topic_variable))
        assert topic_variable in [TOPIC_CONVERSATION, TOPIC_RANDOM, TOPIC_SUBREDDIT], \
            ValueError('topic_variable received unexpected value {} ...'.format(topic_variable))
        # training data
        train_total = math.floor(self.nbr_triples_to_extract * self.train_ratio)
        logging.info("Size first train authors {}".format(len(self.train_authors)))
        self.train_data = self.generate_tasks(self.train_authors, self.train_superset, train_total, topic_variable,
                                              base_task_df=train_data)
        logging.info("  adapted size first train authors {}".format(len(self.train_authors)))

        dev_total = math.floor(self.nbr_triples_to_extract * self.dev_ratio)
        logging.info("Size first dev authors {}".format(len(self.dev_authors)))
        self.dev_data = self.generate_tasks(self.dev_authors, self.dev_superset, dev_total, topic_variable,
                                            base_task_df=dev_data)
        logging.info("  adapted size first dev authors {}".format(len(self.dev_authors)))

        test_total = math.floor(self.nbr_triples_to_extract * self.test_ratio)
        logging.info("Size first test authors {}".format(len(self.test_authors)))
        self.test_data = self.generate_tasks(self.test_authors, self.test_superset, test_total, topic_variable,
                                             base_task_df=test_data)
        logging.info("  adapted size first test authors {}".format(len(self.test_authors)))
        return self.train_data, self.dev_data, self.test_data

    def _init_authors(self):
        """
            split into non-overlapping train, dev and test author set
                via a superset and a first author set per split,
                    anchor sentence always has to be produced from the first author set, while
                    utterance 2 can also be sampled from the superset
        :return:
        """
        # init, where we select a general superset of all possible authors
        #      which is a list of speaker_ids that occur at least self.min_valid_utts times in conversations with others
        #      sets self.nbr_triples_to_extract to the nbr_triples_to_extract number of triples that will be extracted

        assert (self.train_ratio + self.test_ratio + self.dev_ratio == 1)

        # set train, dev, test SUPERSET (authors with at least one (valid) utterance, i.e., not in SKIP_COMMENTS)
        self._set_supersets()

        # Pre-select FIRST AUTHORS (from which the anchors will later be sampled)
        self._preset_firstauthors()

        # calculate the number of maximally possible 'Author over Conversation' Tasks as
        #   FIRST_AUTHORS * (min_valid_utt choose 2)  = nbr first authors * possibilities to choose 2 distinct utt
        #       which corresponds to the number of as this is the number of DIFFERENT first author pairs possible,
        #       which is the limiting factor in triplet generation (superset_first_author is way bigger)
        max_total = (len(self.train_authors) + len(self.test_authors) + len(self.dev_authors)) * \
                    math.comb(self.min_valid_utts, 2)  # before: (self.min_valid_utts - 1)
        logging.info('Maximum possible Tasks: {}'.format(max_total))

        if not self.nbr_triples_to_extract or self.nbr_triples_to_extract > max_total:
            self.nbr_triples_to_extract = max_total
            logging.warning(
                "Given nbr_triples_to_extract number of triples too large, setting nbr_triples_to_extract to {}"
                    .format(self.nbr_triples_to_extract))
        else:
            logging.info("Tasks to be extracted set to {}".format(self.nbr_triples_to_extract))

        # ASSERT that the supersets are supersets
        #   and that there is no overlaps between supersets, i.e., with that also not between first author sets
        assert (not set(self.dev_superset) & set(self.test_superset))
        assert (not set(self.dev_superset) & set(self.train_superset))
        assert (set(self.dev_authors) & set(self.dev_superset) == set(self.dev_authors))
        assert (not set(self.train_superset) & set(self.test_superset))
        assert (not set(self.train_superset) & set(self.dev_superset))
        assert (set(self.train_authors) & set(self.train_superset) == set(self.train_authors))
        assert (not set(self.test_superset) & set(self.train_superset))
        assert (not set(self.test_superset) & set(self.dev_superset))
        assert (set(self.test_authors) & set(self.test_superset) == set(self.test_authors))

    def save_data_split(self, output_dir="", topic_variable: str = TOPIC_CONVERSATION, years: List[int] = SAMPLE_YEARS,
                        author_data_fname: str = 'author_data.json'):
        assert topic_variable in [TOPIC_CONVERSATION, TOPIC_RANDOM, TOPIC_SUBREDDIT]

        file_end = "_subreddits-{}_year-{}-{}_tasks-{}_topic_variable-{}.tsv" \
            .format(len(self.convokit_keys), years[0], years[-1], self.nbr_triples_to_extract, topic_variable)
        train_filename = output_dir + "/train-{}_{}".format(len(self.train_data), file_end)
        logging.info('Saving train data file to {}'.format(train_filename))
        self.train_data.to_csv(train_filename, sep="\t")
        dev_filename = output_dir + "/dev-{}_{}".format(len(self.dev_data), file_end)
        logging.info('Saving dev data file to {}'.format(dev_filename))
        self.dev_data.to_csv(dev_filename, sep="\t")
        test_filename = output_dir + "/test-{}_{}".format(len(self.test_data), file_end)
        logging.info('Saving test data file to {}'.format(test_filename))
        self.test_data.to_csv(test_filename, sep="\t")
        logging.info('Saving train sets to {}')
        author_dict = {
            'train authors': self.train_authors, 'train superset': self.train_superset,
            'dev authors': self.dev_authors, 'dev superset': self.dev_superset,
            'test authors': self.test_authors, 'test superset': self.test_superset}
        with open(output_dir + '/' + author_data_fname, 'w') as fp:
            json.dump(author_dict, fp, indent=4)

    def load_authors_set_from_json(self, data_dir: str, filename: str = 'author_data.json'):
        with open(data_dir + '/' + filename, 'r') as fp:
            author_dict = json.load(fp)
        self.train_authors = author_dict['train authors']
        self.train_superset = author_dict['train superset']
        self.dev_authors = author_dict['dev authors']
        self.dev_superset = author_dict['dev superset']
        self.test_authors = author_dict['test authors']
        self.test_superset = author_dict['test superset']

    def generate_tasks(self, task_authors, task_superset, task_total, topic_variable: str = TOPIC_CONVERSATION,
                       base_task_df=None) -> pd.DataFrame:
        """
            generate a dataframe of task_total tasks
                for the given set of authors and superset authors where authors is a subset of the superset.
        :param task_authors:
        :param task_superset:
        :param task_total:
        :param topic_variable:
        :param base_task_df:
        :return:
        """
        if base_task_df is not None:
            assert task_total == len(base_task_df)
        logging.info('Generating {} Tasks'.format(task_total))
        task_dict = {ANCHOR_COL: [], U1_COL: [], U2_COL: [], ID_A_COL: [], ID_U1_COL: [], ID_U2_COL: [],
                     AUTHOR_A_COL: [], AUTHOR_U1_COL: [], AUTHOR_U2_COL: [], CONVERSATION_A_COL: [],
                     CONVERSATION_U1_COL: [], CONVERSATION_U2_COL: [], SUBREDDIT_A_COL: [], SUBREDDIT_U1_COL: [],
                     SUBREDDIT_U2_COL: [], SAME_AUTHOR_AU1_COL: []}
        del_authors = 0
        for i in range(
                task_total):
            if i % 100 == 0:
                logging.info("At task {}".format(i))
            if base_task_df is None:
                a_id = None
                sa_id = None
            else:
                a_id = base_task_df.iloc[i][ID_A_COL]
                sa_is_u1 = True
                if base_task_df.iloc[i][SAME_AUTHOR_AU1_COL] == 1:
                    sa_id = base_task_df.iloc[i][ID_U1_COL]
                else:
                    sa_is_u1 = False
                    sa_id = base_task_df.iloc[i][ID_U2_COL]
            a, sa, a, da, cur_del_authors = self.random_utterance_quartet(task_authors, task_superset, topic_variable,
                                                                          a_id=a_id, sa_id=sa_id)

            while len([1 for i in range(len(task_dict[ANCHOR_COL]))
                       if (task_dict[ID_A_COL][i] == a.id) and
                          (task_dict[ID_U1_COL][i] == sa.id or task_dict[ID_U2_COL][i] == sa.id)]) > 0:
                # make sure no pair of utterances has appeared before
                a, sa, a, da, cur_del_authors = self.random_utterance_quartet(task_authors, task_superset,
                                                                              topic_variable, a_id=a_id, sa_id=sa_id)
            del_authors += cur_del_authors
            u_order = [sa, da]
            if base_task_df is None:
                # only randomly decide triple task order if it has not been pre-decided
                #   on a previous same conversation triple
                random.shuffle(u_order)
            else:
                if not sa_is_u1:
                    u_order = [da, sa]
            #   LABEL is 1 if same author
            label = 1 if u_order[0] == sa else 0
            # add text, REPLACING '\t' with '    ', '\r' with ''
            update_dict = {ANCHOR_COL: self.remove_illegal_symbols(a.text),
                           U1_COL: self.remove_illegal_symbols(u_order[0].text),
                           U2_COL: self.remove_illegal_symbols(u_order[1].text),
                           ID_A_COL: a.id, ID_U1_COL: u_order[0].id, ID_U2_COL: u_order[1].id,
                           AUTHOR_A_COL: a.speaker.id, AUTHOR_U1_COL: u_order[0].speaker.id,
                           AUTHOR_U2_COL: u_order[1].speaker.id,
                           CONVERSATION_A_COL: a.conversation_id,
                           CONVERSATION_U1_COL: u_order[0].conversation_id,
                           CONVERSATION_U2_COL: u_order[1].conversation_id,
                           SUBREDDIT_A_COL: a.get_conversation().meta['subreddit'],
                           SUBREDDIT_U1_COL: u_order[0].get_conversation().meta['subreddit'],
                           SUBREDDIT_U2_COL: u_order[1].get_conversation().meta['subreddit'],
                           SAME_AUTHOR_AU1_COL: label}
            for key, value in update_dict.items():
                task_dict[key].append(value)
        task_data = pd.DataFrame.from_dict(task_dict)
        assert(len(task_data) == len(task_dict[SAME_AUTHOR_AU1_COL]))
        self.print_summary(task_data, task_authors, task_superset)
        return task_data

    def remove_illegal_symbols(self, text):
        # replace '\t' with '    ', '\r' with ''
        #   otherwise read_csv will encounter errors
        tab_symbol = '\t'
        whitespace_tab = '    '
        carriage_return = '\r'
        empty_str = ''
        text = text.replace(tab_symbol, whitespace_tab).replace(carriage_return, empty_str)
        return text

    @staticmethod
    def print_summary(task_data, task_authors, task_superset):
        """

        :param task_data:
        :param task_authors: first author set
        :param task_superset: superset author set
        :return:
        """
        logging.info('Generated a total of {} tasks consisting of {} distinct authors with {} distinct utterances'
                     .format(len(task_data),
                             len(pd.unique(task_data[[AUTHOR_A_COL, AUTHOR_U1_COL, AUTHOR_U2_COL]].values.ravel('K'))),
                             len(pd.unique(task_data[[ID_A_COL, ID_U1_COL, ID_U2_COL]].values.ravel('K')))))
        logging.info('  Nmber of first authors {} and size of total superset {}'
                     .format(len(task_authors), len(task_superset)))
        logging.info('  Maximum occurring first author is {} with a count of {}'
                     .format(task_data[[AUTHOR_A_COL]].value_counts().idxmax(),
                             task_data[[AUTHOR_A_COL]].value_counts().max()))
        nbr_same_conv = len(task_data[
                                (task_data[CONVERSATION_A_COL] == task_data[CONVERSATION_U1_COL])
                                & (task_data[SAME_AUTHOR_AU1_COL] == 1)
                                ]) + \
                        len(task_data[
                                (task_data[CONVERSATION_A_COL] == task_data[CONVERSATION_U2_COL])
                                & (task_data[SAME_AUTHOR_AU1_COL] == 0)
                                ])
        logging.info('  Share of tasks, where same author utterances came from the same conversation is {0}={1}/{2}'
                     .format(nbr_same_conv / len(task_data), nbr_same_conv, len(task_data)))
        nbr_distinct_same_conv = len(task_data[
                                         ((task_data[CONVERSATION_A_COL] == task_data[CONVERSATION_U1_COL]) &
                                          (task_data[SAME_AUTHOR_AU1_COL] == 0)) |
                                         ((task_data[CONVERSATION_A_COL] == task_data[CONVERSATION_U2_COL]) &
                                          (task_data[SAME_AUTHOR_AU1_COL] == 1))
                                         ])
        logging.info('  Share of tasks, where distinct author utterances came from the same conversation is '
                     '{0}={1}/{2}'
                     .format(nbr_distinct_same_conv / len(task_data), nbr_distinct_same_conv, len(task_data)))
        nbr_da_subset = len(task_data[
                                ((task_data[AUTHOR_U2_COL].isin(task_authors)) &
                                 (task_data[SAME_AUTHOR_AU1_COL] == 1)) |
                                ((task_data[AUTHOR_U1_COL].isin(task_authors)) &
                                 (task_data[SAME_AUTHOR_AU1_COL] == 0))
                                ])
        logging.info('  Share of DA utterances, where utterance was written by a subset author: {0}={1}/{2}'
                     .format(nbr_da_subset / len(task_data), nbr_da_subset, len(task_data)))
        nbr_same_sub = len(task_data[
                                (task_data[SUBREDDIT_A_COL] == task_data[SUBREDDIT_U1_COL])
                                & (task_data[SAME_AUTHOR_AU1_COL] == 1)
                                ]) + \
                        len(task_data[
                                (task_data[SUBREDDIT_A_COL] == task_data[SUBREDDIT_U2_COL])
                                & (task_data[SAME_AUTHOR_AU1_COL] == 0)
                                ])
        logging.info('  Share of tasks, where same author utterances came from the same subreddit: {}={}/{}'
                     .format(nbr_same_sub/len(task_data), nbr_same_sub, len(task_data)))
        nbr_distinct_same_sub = len(task_data[
                                        ((task_data[SUBREDDIT_A_COL] == task_data[SUBREDDIT_U1_COL]) &
                                         (task_data[SAME_AUTHOR_AU1_COL] == 0)) |
                                        ((task_data[SUBREDDIT_A_COL] == task_data[SUBREDDIT_U2_COL]) &
                                         (task_data[SAME_AUTHOR_AU1_COL] == 1))
                                        ])
        logging.info('  Share of tasks, where distinct author utterances came from the same subreddit: {}={}/{}'
                     .format(nbr_distinct_same_sub/len(task_data), nbr_distinct_same_sub, len(task_data)))

    def random_utterance_quartet(self, first_authors, superset_authors, topic_variable=TOPIC_CONVERSATION,
                                 a_id=None, sa_id=None):
        """
        Randomly select two utterance pairs: first by same author and second by distinct authors;
        with sa_utt and u3 being the same utterance by the same author -- i.e., those are rather triples

        ! CURRENTLY: it could be this runs into an endless loop, if there exists an author in the
        first_authors set that does not fulfill the validity condition of at least appearing
        in one conversation with another author from the superset ... so far this has never happened and seems unlikely

        :param topic_variable:
        :param first_authors: authors to sample utterances from
        :param superset_authors: exclude those authors
        :param non_empty: ?
        :param a_id: convokit id for the a utterance already given
        :param sa_id: convokit id for the sa utterance already given
        :return: (a, sa_utt), (a, da_utt) in form of utterance objects and how many first authors were removed
        """
        if topic_variable not in [TOPIC_CONVERSATION, TOPIC_SUBREDDIT, TOPIC_RANDOM]:
            raise ValueError('topic variable has unknown value {}'.format(topic_variable))

        given_sa_pair = False
        if a_id is not None or sa_id is not None:
            given_sa_pair = True
            assert a_id is not None and sa_id is not None
            # ATTENTION: given a (A, SA)-pair this assumes that we are not using the topic variable same conversation !
            assert not topic_variable == TOPIC_CONVERSATION, \
                ValueError('Unexpected topic variable conversation was given ...')

        removed_first_authors = 0
        if not given_sa_pair:

            # RANDOM DISTINCT author pair (A, DA)
            #   1. randomly select first author of utterance A
            speaker_a = self.corpus.get_speaker(random.choice(first_authors))
            #       randomly select a nonempty utterance A by speaker
            a_id = self.random_nonempty_utt_by_speaker(speaker_a)

        else:  # with GIVEN (A, SA)-pair

            speaker_a = self.corpus.get_utterance(a_id).speaker

            # sample random DA utterance from superset authors (similar to same sampling)
            # speaker_a = self.corpus.get_utterance(a_id).speaker
            # speaker_da = self.select_random_distinct_author(speaker_a, superset_authors)
            # da_id = self.random_nonempty_utt_by_speaker(speaker_da)

        #   2. randomly select utterance DA
        if topic_variable == TOPIC_CONVERSATION:  # A and DA have to come from the SAME CONVERSATION
            nbr_utt_by_a = len(speaker_a.get_utterance_ids())
            # make sure the utterance appears in a valid conversation, i.e.,
            # where at least one other superset author participated in with a non-empty utterance
            da_id = self.get_same_conv_superset_utt(a_id, self.corpus, superset_authors)
            nbr_tried_utt = 0
            while da_id is False and nbr_tried_utt < min(math.floor(nbr_utt_by_a / self.min_valid_utts),
                                                         MAX_TRY_UTT):
                # while nbr_tried_utt < min(nbr_utt_by_a, MAX_TRY_UTT) and len(superset_utts) == 0:
                #    a_id = self.random_nonempty_utt_by_speaker(speaker_a)
                da_id = self.get_same_conv_superset_utt(a_id, self.corpus, superset_authors)
                nbr_tried_utt += 1
            if da_id is False or nbr_tried_utt == min(math.floor(nbr_utt_by_a / self.min_valid_utts), MAX_TRY_UTT):
                first_authors.remove(speaker_a.id)
                removed_first_authors += 1
                logging.warning(" Removed first author {}".format(speaker_a.id))
                sa_utt, da_utt, u3, u4, cur_del_authors = self.random_utterance_quartet(first_authors,
                                                                                        superset_authors,
                                                                                        topic_variable)
                return sa_utt, da_utt, u3, u4, cur_del_authors + removed_first_authors
        elif topic_variable == TOPIC_SUBREDDIT:  # A and DA have to come from the SAME SUBREDDIT
            da_id = None
            subreddit_a = self.corpus.get_utterance(a_id).meta[TOPIC_SUBREDDIT]
            while da_id is None:
                speaker_da = self.select_random_distinct_author(speaker_a, superset_authors)
                da_suba_utt_ids = speaker_da.get_utterance_ids(
                    lambda utt: utt.meta[TOPIC_SUBREDDIT] == subreddit_a and TaskGenerator.is_valid(utt.text))
                nbr_utt_by_da = len(da_suba_utt_ids)
                if nbr_utt_by_da > 0:
                    da_id = random.choice(da_suba_utt_ids)
                    if pd.isna(self.corpus.get_utterance(da_id)):
                        raise ValueError("Tried to add a non valid utterance for distinct author {}".format(da_id))
        else:  # A and DA do not have to come from the SAME CONVERSATION or the SAME SUBREDDIT
            assert topic_variable == TOPIC_RANDOM, AssertionError('Topic Variable has an unexpected value')
            speaker_da = self.select_random_distinct_author(speaker_a, superset_authors)
            da_id = self.random_nonempty_utt_by_speaker(speaker_da)

        if not given_sa_pair:
            # only select RANDOM SAME author pair (A, SA) WHEN this has not been pre-selected
            speaker_sa = speaker_a
            sa_id = self.random_nonempty_utt_by_speaker(speaker_sa, unequal_to=a_id)

        return self.corpus.get_utterance(a_id), self.corpus.get_utterance(sa_id), \
               self.corpus.get_utterance(a_id), self.corpus.get_utterance(da_id), removed_first_authors

    def select_random_distinct_author(self, speaker_a, superset_authors):
        speaker_da = self.corpus.get_speaker(random.choice(superset_authors))
        while speaker_da == speaker_a:
            speaker_da = self.corpus.get_speaker(random.choice(superset_authors))
        return speaker_da

    @staticmethod
    def get_same_conv_superset_utt(utterance_id, corpus, superset_authors):
        """
        Return a nonempty utterances written by superset_authors\{utterance author}
        that appears in the same conversation as utterance_id. returns False if no such utterance exists.
        :param utterance_id: id for an utterance
        :param corpus: convokit corpus object
        :param superset_authors: set of authors
        :return:
        """
        utterance = corpus.get_utterance(utterance_id)
        conversation = utterance.get_conversation()
        speaker = utterance.get_speaker()

        possible_speakers = list(set(conversation.get_speaker_ids()).intersection(set(superset_authors)))

        # valid_superset_utterances = [utt.id for utt in conversation.iter_utterances(
        #    lambda u: u.speaker.id != speaker.id and u.speaker.id in superset_authors and u.text not in SKIP_COMMENTS
        # )]

        # valid_superset_utterances = []
        utt_ids = conversation.get_utterance_ids()
        random.shuffle(utt_ids)
        for utt_id in utt_ids:
            utt = corpus.get_utterance(utt_id)
            if utt.speaker.id == speaker.id:
                continue
            not_valid = not TaskGenerator.is_valid(utt.text)
            if not_valid:
                continue
            if utt.speaker.id not in possible_speakers:
                continue
            return utt_id
            # valid_superset_utterances.append(utt_id)
        return False

        # return valid_superset_utterances

    @staticmethod
    def is_valid(utt: str) -> bool:
        if type(utt) != str or \
                utt is None or \
                utt in REDDIT_SKIP_COMMENTS or \
                utt.isspace():
            return False
        else:
            return True

    @staticmethod
    def random_nonempty_utt_by_speaker(s1, unequal_to=None):
        """
            Get a valid utterances for the given speaker s1
                valid, i.e., non empty or deleted or removed
        :param unequal_to: optional - id of an utterance that should not be selected
        :param s1: speaker
        :return: id of the utterance
        """
        u1_id = random.choice(s1.get_utterance_ids(
            lambda utt: TaskGenerator.is_valid(utt.text) and utt.id != unequal_to
        ))
        return u1_id

    def _preset_firstauthors(self):
        #   from the supersets select those that have at least
        #   self.min_valid_utts + 10% from self.min_valid_utts non-empty comments
        #   this is ASSUMING that
        #       this is a well enough approximation that those authors work as first authors most of the time,
        #       i.e., appear at least self.min_valid_utts with another author in the same conversation
        #       from the same superset
        logging.info("With requested number of at least self.min_valid_utts={} utterances extractable per author "
                     "and after filtering out authors with less than "
                     "self.min_valid_utts + max(1, round(0.1*self.min_valid_utts)) non-empty utterances "
                     .format(self.min_valid_utts))
        self.train_authors = [speaker_id for speaker_id in self.train_superset
                              if len(self.corpus.get_speaker(speaker_id).get_utterance_ids(
                lambda u: TaskGenerator.is_valid(u.text))) >=
                              self.min_valid_utts + max(1, round(0.1 * self.min_valid_utts))]
        logging.info("  {} possible train_author candidates remain.".format(len(self.train_authors)))
        self.test_authors = [speaker_id for speaker_id in self.test_superset
                             if len(self.corpus.get_speaker(speaker_id).get_utterance_ids(
                lambda u: TaskGenerator.is_valid(u.text))) >=
                             self.min_valid_utts + max(1, round(0.1 * self.min_valid_utts))]
        logging.info("  {} possible test_author candidates remain.".format(len(self.test_authors)))
        self.dev_authors = [speaker_id for speaker_id in self.dev_superset
                            if len(self.corpus.get_speaker(speaker_id).get_utterance_ids(
                lambda u: TaskGenerator.is_valid(u.text))) >=
                            self.min_valid_utts + max(1, round(0.1 * self.min_valid_utts))]
        logging.info("  {} possible dev_author candidates remain.".format(len(self.dev_authors)))

    def _set_supersets(self):
        superset_authors = [speaker.id
                            for speaker in self.corpus.iter_speakers(
                lambda s: len(
                    s.get_utterance_ids(
                        lambda u: TaskGenerator.is_valid(u.text))
                ) > 0
            )]
        #   SPLIT superset 70 - 15 - 15 and shuffle dataset during splitting
        self.train_superset, self.test_superset = sklearn.model_selection.train_test_split(superset_authors,
                                                                                           train_size=self.train_ratio,
                                                                                           shuffle=True)
        tmp_train_size = self.dev_ratio / (1 - self.train_ratio)
        self.dev_superset, self.test_superset = sklearn.model_selection.train_test_split(self.test_superset,
                                                                                         train_size=tmp_train_size,
                                                                                         shuffle=True)
        del superset_authors
        logging.info(" TRAIN ratio at " + str(self.train_ratio) + " TEST ratio at " + str(self.test_ratio)
                     + " DEV ratio at " + str(self.dev_ratio))

    # NOT USED --> changed min_valid_utts to mean at least these number of non-empty utterances
    @staticmethod
    def extract_first_authors(superset_authors, corpus, lower_thresh):
        """
        Extract first authors from the superset of authors. Those need to have at least min_valid_utts valid utterances
        :param superset_authors: superset of authors out of which the first authors are chosen
        :param corpus: conovokit corpus object
        :param lower_thresh: lower threshold of at least that many valid utterances to become a first author
        :return:
        """
        # corpus = self.corpus
        # min_valid_utts = self.min_valid_utts

        # Only select those as FIRST AUTHOR that have at least X utterances with others in the same superset
        first_authors = []
        #   Select a subset of the possible SET_AUTHORS
        for speaker_id in superset_authors:
            #   ONLY select first authors that have at least min_valid_utts number of non-empty utterances
            if len(corpus.get_speaker(speaker_id).get_utterance_ids(lambda u: TaskGenerator.is_valid(u.text))) \
                    < lower_thresh:
                continue

            #   ONLY count valid utterances
            cur_valid_utts = []
            for utt_id in corpus.get_speaker(speaker_id).get_utterance_ids():
                #   VALID utterances are: - not empty
                if not TaskGenerator.is_valid(corpus.get_utterance(utt_id).text):
                    continue
                #                         - with at least 1 distinct author from the superset ocurring in the conv
                #                           with a non-empty utterance
                # TODO: could do with X other authors from the superset, where X is the number of utt. by that author in conv
                cur_conv = corpus.get_utterance(utt_id).get_conversation()
                for cur_conv_utt in cur_conv.iter_utterances():
                    # cur_conv_utt = corpus.get_utterance(conv_utt_id)
                    if cur_conv_utt.speaker.id != speaker_id and TaskGenerator.is_valid(cur_conv_utt.text) \
                            and cur_conv_utt.speaker.id in superset_authors:
                        cur_valid_utts.append(utt_id)

            if len(cur_valid_utts) >= lower_thresh:  # Condition of at least min_valid_utts valid utterances
                first_authors.append(speaker_id)
                if len(first_authors) % 100 == 0:
                    logging.info("at first author number " + str(len(first_authors)))

        return first_authors


