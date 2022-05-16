"""
    generate the CONVERSATION/DOMAIN/RANDOM topic proxy split for the (contrastive) AV task on the
        convokit (https://convokit.cornell.edu/) Reddit data
"""

import logging
import sys, os
import argparse
import time
from global_identifiable import OUTPUT_FOLDER, CONVO_CACHE
from global_const import SUB_LIST, SAMPLE_YEARS, MIN_VALID_UTTS, TOTAL, CONVS_PER_SUB, set_logging, \
    set_global_seed, TOPIC_SUBREDDIT, TOPIC_RANDOM, TOPIC_CONVERSATION

sys.path.append(os.path.join('..', 'style_embed/utility'))
from convokit_generator import TaskGenerator, ConvokitStream
set_logging()


def main(convokit_keys=SUB_LIST, min_valid_utts=MIN_VALID_UTTS, total=TOTAL,
         output_dir=OUTPUT_FOLDER, convo_directory=CONVO_CACHE,
         debug=False, d_only=False, years=SAMPLE_YEARS, convs_per_subreddit=CONVS_PER_SUB):
    """
        download convokit data and generate CAV tasks from the convokit_keys subreddits
        (this only works with the Reddit convokit keys)

        :param convokit_keys: which convokit reddit communities to look at
        :param min_valid_utts: minimum number of valid utterances produced by an author
        :param total: number of tasks to generate
        :param output_dir: where to save the data to
        :param convo_directory: folder convokit cache
        :param debug:
        :param d_only: whether to only download the conovkit dataset or already generate tasks
        :param years: which Reddit years to consider
        :param convs_per_subreddit: number of unique conversations considered per subreddit
    """
    if d_only:
        logging.info("Set to download only ...")
        ConvokitStream(convokit_data_keys=convokit_keys, directory=convo_directory, years=years, d_only=d_only,
                       convo_per_sub=convs_per_subreddit)
        return
    start = time.time()
    set_global_seed()
    if len(convokit_keys) > 1 or 'sample' not in convokit_keys[0]:
        task_gen = TaskGenerator(convokit_data_keys=convokit_keys, total=total, min_valid_utts=min_valid_utts,
                                 directory=convo_directory, convo_per_sub=convs_per_subreddit)
    else:
        # already filtered and sampled before, i.e., no need to sample conversations or authors/convos with too few utts
        logging.info('Loading corpus from previously filtered and sampled data {}'
                     .format(convo_directory + "/" + convokit_keys[0]))
        task_gen = TaskGenerator(convokit_data_keys=convokit_keys, total=total, directory=convo_directory,
                                 convo_per_sub=None)
    conv_train, conv_dev, conv_test = task_gen._get_data_split(topic_variable=TOPIC_CONVERSATION)
    task_gen.save_data_split(output_dir, topic_variable=TOPIC_CONVERSATION)
    end_conv = time.time()
    logging.info('Time for sameconv: {}'.format(end_conv-start))

    sub_train, sub_dev, sub_test = task_gen._get_data_split(topic_variable=TOPIC_SUBREDDIT,
                                                            train_data=conv_train, dev_data=conv_dev,
                                                            test_data=conv_test)
    task_gen.save_data_split(output_dir, topic_variable=TOPIC_SUBREDDIT)
    if not debug:
        del sub_train, sub_dev, sub_test
    end_sub = time.time()
    logging.info('Time for samesub: {}'.format(end_sub - end_conv))

    rand_train, rand_dev, rand_test = task_gen._get_data_split(topic_variable=TOPIC_RANDOM, train_data=conv_train,
                                                               dev_data=conv_dev, test_data=conv_test)
    task_gen.save_data_split(output_dir, topic_variable=TOPIC_RANDOM)
    if not debug:
        del rand_train, rand_dev, rand_test
    logging.info('Time for rand: {}'.format(time.time() - end_sub))

    if debug:
        return {
            "conversation": [conv_train, conv_dev, conv_test],
            "subreddit": [sub_train, sub_dev, sub_test],
            "random": [rand_train, rand_dev, rand_test]
        }


if __name__ == "__main__":
    """
        example call: 
        python3 /home/uu_cs_nlpsoc/awegmann/digital-society_intergenerational-empathy/src/style_sim/generate_dataset.py 
            -out '/hpc/uu_cs_nlpsoc/02-awegmann/style_discovery/train_data' 
            -total 300000  
            -convoset "merged-sample"

    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Generating Datasets for (1) Training and Evaluation or (2) Validation.')
    parser.add_argument('-convoset', '--convokit_dataset',
                        default=SUB_LIST,
                        nargs="*",
                        type=str,
                        help="For example 'reddit-corpus-small' or 'subreddit-politics' or 'wiki-corpus'")
    parser.add_argument('-low', '--min_valid_utts', default=MIN_VALID_UTTS,
                        help="the lower threshold for the minimum number of utterances that"
                             " a possible first author must have written")
    parser.add_argument('-total', '--nbr_triples_to_extract', default=TOTAL,
                        help="Total number of utterances that should be extracted. Should not change this.")
    parser.add_argument('-out', '--output_dir', default=OUTPUT_FOLDER,
                        help="path to output directory for dataset")
    parser.add_argument('-donly', '--download_only', dest='download_only', action='store_true',  # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
                        help="whether to only download and filter the datasets. Default is False.")
    parser.add_argument('-no-donly', '--no-download_only', dest='download_only', action='store_false')
    parser.set_defaults(download_only=False)

    args = parser.parse_args()
    logging.info("Using convokit keys {}".format(args.convokit_dataset))
    main(convokit_keys=args.convokit_dataset, min_valid_utts=int(args.min_valid_utts), total=int(args.nbr_triples_to_extract), output_dir=args.output_dir,
         d_only=args.download_only)  # , au2=args.au2same)
