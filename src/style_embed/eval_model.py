"""
    evaluate the trained models -- for example, directly called after fine-tuning
"""
import sys
import os
import argparse
import logging

from global_identifiable import OUTPUT_FOLDER, set_cache
from global_const import HPC_DEV_DATASETS, HPC_TEST_DATASETS, get_complete_model_name_from_path, \
    get_results_folder, set_global_seed, set_logging

set_global_seed()
set_cache()
sys.path.append(os.path.join('..', 'utility'))
# include module from utility directory
from trained_similarities import TunedSentenceBertSimilarity

# this needs the STEL project, needs to be accessible for the project to include
from STEL_Or_Content import test_model_on_STEL

SBert = "SBert"


def main(model_path='../../../models/', test_files=HPC_DEV_DATASETS, test_stel=True, test_AV=True):
    """
        evaluate model that is saved in model_path on test_files
        :param model_path: path to model
        :param test_files: path to contrastive AV tasks that are tested, usually dev or test tasks
        :param test_stel: whether STEL and STEL-Or-Content should be tested (needs proprietary data & more runtime)
        :param test_AV: whether to test on contrastive AV task and non-contrastive AV tasks
    """

    set_logging()
    model = TunedSentenceBertSimilarity(model_path)
    _evaluate_model(model, model_path, test_files, test_stel, test_AV)


def _evaluate_model(model, model_path, test_files, test_stel, test_AV):
    """
        see main
    """
    model_name = get_complete_model_name_from_path(model_path)
    results_folder = get_results_folder(model_path=model_path)
    if test_stel:
        test_model_on_STEL(model, model_name, results_folder)

    if test_AV:
        # test on dev set
        logging.info("testing model on AV task ... ")
        logging.info(" ... testing on files {}".format(test_files))
        from evaluation_metrics import triple_test_sim_function
        for test_file in test_files:
            logging.info("testing on {} ...".format(test_file))
            triple_test_sim_function(similarity_function_callable=model.similarities,
                                     triple_task_filename=test_file, output_folder=results_folder,
                                     sim_function_name=model_name)


if __name__ == "__main__":
    set_logging()
    parser = argparse.ArgumentParser(description='Evaluating a Neural Model.')
    parser.add_argument('-mt', '--model_type', default="Bert",
                        help='model type: Bert or SentenceBert')  # dest='accumulate', action='store_const',
    parser.add_argument('-md', '--model_dir', default=None, help='path to trained model, '
                                                                 'do not refer directly to the bin file')
    parser.add_argument('-name', '--model_name', default="", help='model name to be used in plots')
    # parser.add_argument('-test', '--test_dataset', default=None, help="path to dataset "
    #                                                                   "that the triple task should be tested on")
    parser.add_argument('-out', "--model_output_dir", default=OUTPUT_FOLDER, help="path to eval output")

    parser.add_argument('-stel', '--stel', dest='stel', action='store_true',
                        # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
                        help="whether STEL should be tested for the model")
    parser.add_argument('-no-stel', '--no-stel', dest='stel', action='store_false')
    parser.set_defaults(stel=True)

    parser.add_argument('-ttask', '--ttask', dest='ttask', action='store_true',
                        # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
                        help="whether triple task should be tested for the model")
    parser.add_argument('-no-ttask', '--no-ttask', dest='ttask', action='store_false')
    parser.set_defaults(ttask=True)

    parser.add_argument('-test', '--test_set', dest='test', action='store_true',
                        # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
                        help="whether triple task should be tested for the model")
    parser.add_argument('-dev', '--dev_set', dest='test', action='store_false')
    parser.set_defaults(test_set=True)

    # parser.add_argument('-seed', '--seed', default=SEED)

    args = parser.parse_args()
    logging.info("Working with output folder {}".format(args.model_output_dir))

    if args.test:
        logging.info("testing on test dataset ...")
        test_set = HPC_TEST_DATASETS
    else:
        logging.info("testing on dev dataset ...")
        test_set = HPC_DEV_DATASETS

    main(model_path=args.model_dir, test_stel=args.stel, test_AV=args.ttask, test_files=test_set)
