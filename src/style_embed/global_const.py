import os

import random

import logging
import numpy
import sys

from global_identifiable import TRAIN_DATA_BASE, RESULTS_FOLDER

SEED = 1404
BERT_CASED_BASE_MODEL = "bert-base-cased"
BERT_UNCASED_BASE_MODEL = "bert-base-uncased"

# Column names for Training tasks
SAME_AUTHOR_AU1_COL = 'Same Author Label'  # label is 1 if A and U1 are written by the same author, 0 otherwise
SUBREDDIT_U2_COL = 'Subreddit U2'
SUBREDDIT_U1_COL = 'Subreddit U1'
SUBREDDIT_A_COL = 'Subreddit A'
CONVERSATION_U2_COL = 'Conversation ID U2'
CONVERSATION_U1_COL = 'Conversation ID U1'
CONVERSATION_A_COL = 'Conversation ID A'
ID_U2_COL = 'ID U2'
ID_U1_COL = 'ID U1'
ID_A_COL = 'Utterance ID A'
AUTHOR_U2_COL = 'Author U2'
AUTHOR_U1_COL = 'Author U1'
AUTHOR_A_COL = 'Author A'
U2_COL = 'Utterance 2 (U2)'
U1_COL = 'Utterance 1 (U1)'
ANCHOR_COL = 'Anchor (A)'

# Topic variables
TOPIC_SUBREDDIT = 'subreddit'
TOPIC_RANDOM = 'random'
TOPIC_CONVERSATION = 'conversation'

SUB_LIST = ["subreddit-motorcycles", "subreddit-Libertarian", "subreddit-conspiracy",
            "subreddit-IAmA", "subreddit-MMA", "subreddit-politics", "subreddit-techsupport", "subreddit-MensRights",
            "subreddit-hockey", "subreddit-canada", "subreddit-australia", "subreddit-business",
            "subreddit-hiphopheads", "subreddit-AskMen", "subreddit-sex", "subreddit-Guildwars2", "subreddit-news",
            "subreddit-relationships", "subreddit-skyrim", "subreddit-Music", "subreddit-askscience",
            "subreddit-electronic_cigarette", "subreddit-SquaredCircle", "subreddit-WTF", "subreddit-OkCupid",
            "subreddit-science", "subreddit-movies", "subreddit-anime", "subreddit-apple", "subreddit-cringepics",
            "subreddit-pokemontrades", "subreddit-Bitcoin", "subreddit-TwoXChromosomes", "subreddit-Christianity",
            "subreddit-todayilearned", "subreddit-POLITIC", "subreddit-offbeat", "subreddit-guns", "subreddit-Android",
            "subreddit-Frugal", "subreddit-pics", "subreddit-DebateReligion", "subreddit-Fitness",
            "subreddit-photography", "subreddit-pokemon", "subreddit-aww", "subreddit-unitedkingdom",
            "subreddit-technology", "subreddit-tf2", "subreddit-gonewild", "subreddit-Minecraft", "subreddit-buildapc",
            "subreddit-funny", "subreddit-Games", "subreddit-AskWomen", "subreddit-wow", "subreddit-worldnews",
            "subreddit-books", "subreddit-fantasyfootball", "subreddit-gaming", "subreddit-videos",
            "subreddit-baseball", "subreddit-AdviceAnimals", "subreddit-Diablo", "subreddit-teenagers",
            "subreddit-cringe", "subreddit-malefashionadvice", "subreddit-explainlikeimfive", "subreddit-cars",
            "subreddit-soccer", "subreddit-asoiaf", "subreddit-leagueoflegends", "subreddit-CFB",
            "subreddit-MakeupAddiction", "subreddit-gifs", "subreddit-DotA2", "subreddit-starcraft", "subreddit-dayz",
            "subreddit-nfl", "subreddit-Random_Acts_Of_Amazon", "subreddit-relationship_advice", "subreddit-Drugs",
            "subreddit-NoFap", "subreddit-programming", "subreddit-atheism", "subreddit-magicTCG", "subreddit-trees",
            "subreddit-Economics", "subreddit-nba", "subreddit-AmItheAsshole", "subreddit-MovieDetails",
            "subreddit-singapore", "subreddit-ShingekiNoKyojin", "subreddit-Naruto", "subreddit-Marvel",
            "subreddit-tifu", "subreddit-rupaulsdragrace", "subreddit-LifeProTips", "subreddit-travel",
            "subreddit-AskReddit"]
SAMPLE_YEARS = [2018]
MIN_VALID_UTTS = 4
TOTAL = 300000  # 30000 = 30.000 * 10
CONVS_PER_SUB = 600  # 50*12
MIN_COM_PER_CONV = 10
UNCASED_NEXT_BERT = "nextbert"
CASED_NEXT_BERT = "NextBert"
CASED_SEQ_BERT = "SeqBert"
UNCASED_S_BERT = "sbert"
CASED_S_BERT = "SBert"
CASED_BASE_BERT = "Bert"
ROBETA_BASE = "RoBERTa"
UNCASED_BASE_BERT = "bert"
MODEL_TYPES = [CASED_SEQ_BERT, CASED_NEXT_BERT, UNCASED_NEXT_BERT, UNCASED_S_BERT, CASED_BASE_BERT, UNCASED_BASE_BERT,
               CASED_S_BERT]

HPC_DEV_DATASETS = [TRAIN_DATA_BASE +
                    '/dev-45000__subreddits-100-2018_tasks-300000__topic-variable-conversation.tsv',
                    TRAIN_DATA_BASE +
                    '/dev-45000__subreddits-100-2018_tasks-300000__topic-variable-subreddit.tsv',
                    TRAIN_DATA_BASE +
                    '/dev-45000__subreddits-100-2018_tasks-300000__topic-variable-random.tsv']
HPC_TEST_DATASETS = [TRAIN_DATA_BASE +
                     '/test-45000__subreddits-100-2018_tasks-300000__topic-variable-conversation.tsv',
                     TRAIN_DATA_BASE +
                     '/test-45000__subreddits-100-2018_tasks-300000__topic-variable-subreddit.tsv',
                     TRAIN_DATA_BASE +
                     '/test-45000__subreddits-100-2018_tasks-300000__topic-variable-random.tsv']


def set_global_seed(seed=SEED, w_torch=True):
    """
    Make calculations reproducible by setting RANDOM seeds
    :param seed:
    :param w_torch:
    :return:
    """
    # set the global variable to the new var throughout
    global SEED
    SEED = seed
    if 'torch' not in sys.modules:
        w_torch = False
    if w_torch:
        import torch
        logging.info(f"Running in deterministic mode with seed {seed}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    numpy.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_logging():
    """
    set logging format for calling logging.info
    :return:
    """
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        stream=sys.stdout)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def set_torch_device():
    import torch
    global device
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def get_complete_model_name_from_path(model_path):
    return f"{os.path.basename(os.path.dirname(model_path))}-{os.path.basename(model_path)}"


def generate_file_prefix(model_path, triple_task_filename):
    """
        Generates a (begin) filename of the form
            dev-45000__subreddits-100-2018_tasks-300000__topic-variable-conversation_
            topic-sub-bert-base-uncased-loss-triplet-margin-0.5-evaluator-triplet

    :param model_path: chosen distinct sim function name, e.g.,
        'topic-variable-conversation_bert-base-uncased-loss-triplet-margin-0.5-evaluator-binary'
    :param triple_task_filename: path to the triple task that is tested atm, e.g.,
        'dev-45000__subreddits-100-2018_tasks-300000__topic-variable-conversation.tsv'
    :return:
    """
    model_name = get_complete_model_name_from_path(model_path)
    file_start = f"{os.path.basename(triple_task_filename).split('.')[0]}_{model_name}"
    return file_start


def get_results_folder(model_path):
    topic_var = os.path.basename(os.path.dirname(model_path))
    w_seed = False
    if len(topic_var) > 10:
        w_seed = True
        seed = f"{os.path.basename(model_path)}"
        model_path = os.path.dirname(model_path)

        topic_var = os.path.basename(os.path.dirname(model_path))

    base_model = UNCASED_BASE_BERT
    if 'bert-base-cased' in model_path:
        base_model = CASED_BASE_BERT
    elif 'roberta-base' in model_path:
        base_model = ROBETA_BASE

    if 'loss-' in model_path:
        loss = os.path.basename(model_path).split('loss-')[1].split("-margin")[0]
        margin = f"margin-{os.path.basename(model_path).split('-margin-')[1].split('-')[0]}"
        if not w_seed:
            return f"{RESULTS_FOLDER}/{topic_var}/{base_model}/{loss}/{margin}"
        else:
            return f"{RESULTS_FOLDER}/{topic_var}/{base_model}/{loss}/{margin}/{seed}"
    else:
        return f"{RESULTS_FOLDER}/base/{base_model}"

