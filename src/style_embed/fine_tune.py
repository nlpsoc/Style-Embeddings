"""
    Script for fine-tuning model via the sentence bert transformers library
"""

import logging
import sys
import os
import argparse
import eval_model
from typing import List

from global_identifiable import OUTPUT_FOLDER, STRANFORMERS_CACHE
from global_const import HPC_DEV_DATASETS, HPC_TEST_DATASETS, set_logging, SEED

sys.path.append(os.path.join('..', 'utility'))
from neural_trainer import SentenceBertFineTuner
from training_const import BINARY_EVALUATOR, CONTRASTIVE_LOSS, BERT_UNCASED_BASE_MODEL, BATCH_SIZE, EPOCHS, MARGIN

set_logging()


def main(train_file: str = None, dev_file: str = None, test_files: List[str] = None, model_type: str = "SBert",
         model_key=BERT_UNCASED_BASE_MODEL, loss=CONTRASTIVE_LOSS, margin=MARGIN, cache_folder=STRANFORMERS_CACHE,
         batch_size=BATCH_SIZE, eval=True, evaluation_type=BINARY_EVALUATOR, epochs=EPOCHS, profile=False, seed=SEED):
    logging.info("Working with model cache folder {}".format(cache_folder))
    if train_file and dev_file:
        logging.info("working with model type " + str(model_type))
        logging.info("training with batch size {}".format(batch_size))
        if model_type == "SBert":
            tuna = SentenceBertFineTuner(model_path=model_key, train_filename=train_file, dev_filename=dev_file,
                                         loss=loss, margin=margin, cache_folder=cache_folder,
                                         evaluation_type=evaluation_type, seed=seed)
        else:
            logging.warning("given model type is not valid ... ")
            return
        logging.info("Given model type " + str(model_type))
        logging.info("Starting training on " + train_file)
        save_dir = tuna.train(epochs=epochs, batch_size=batch_size, profile=profile)

        if eval:
            hyperparameters = os.path.basename(os.path.normpath(save_dir))
            topic_variable = os.path.basename(os.path.dirname(os.path.normpath(save_dir)))
            model_name = topic_variable + "_" + hyperparameters
            logging.info("Evaluating model {}".format(model_name))

            if not test_files:
                test_files = [dev_file]
            eval_model.main(model_path=save_dir, test_files=test_files)  # TASK_BASE

        # SAVING is not necessary as train already saves best model to cache folder

        return tuna


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine Tuning a Neural Model.')
    parser.add_argument('-test', '--test_dataset', default=None, help="path to test dataset")

    parser.add_argument('-mp', '--model_path', default=BERT_UNCASED_BASE_MODEL,
                        help='model path for SentenceBert, also accepts HuggingFace base models. '
                             'Then generates new model with mean pooling.')
    parser.add_argument('-s', '--seed', default=SEED,
                        help=f"set random seed for training. Default is {SEED}")
    parser.add_argument('-ml', '--model_loss', default=CONTRASTIVE_LOSS,
                        help='SentenceBert loss keyword for model training. Default is contrastive.')
    parser.add_argument('-m', '--margin', default=MARGIN,
                        help='Margin value (default is {}) relevant for contrastive and triplet losses'.format(MARGIN))
    parser.add_argument('-eval', '--evaluation_type', default=BINARY_EVALUATOR,
                        help='Whether to use a binary (default) or triplet evaluator on dev set.')
    parser.add_argument('-train', '--train_filename', default=None, help="path to train dataset")
    parser.add_argument('-dev', '--dev_dataset', default=None, help="path to dev dataset")

    parser.add_argument('-ttest', '--ttest', dest='ttest', action='store_true',  # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
                        help="tuned model should be tested on test and not dev set")
    parser.add_argument('-no-ttest', '--no-ttest', dest='ttest', action='store_false')
    parser.set_defaults(ttest=False)

    # parser.add_argument('-out', "--model_output_dir", default=OUTPUT_FOLDER, help="path to output results")
    args = parser.parse_args()
    logging.info("Working with output folder {}".format(args.model_output_dir))

    if args.ttest is False:
        logging.info("TESTING ON DEV DATASET")
        test_filenames = HPC_DEV_DATASETS
    else:
        logging.info("TESTING ON TEST DATASET")
        test_filenames = HPC_TEST_DATASETS

    main(train_file=args.train_filename, dev_file=args.dev_dataset, test_files=test_filenames,
         model_key=args.model_path, loss=args.model_loss, margin=float(args.margin),
         evaluation_type=args.evaluation_type, seed=int(args.seed))

