"""
    similarity implementation, partly copied from STEL project
"""
import logging
from typing import List
from abc import ABC
import torch
from sentence_transformers import SentenceTransformer
EVAL_BATCH_SIZE = 64


class Similarity(ABC):
    """
        copied from https://github.com/nlpsoc/STEL
        Abstract Base similarity class
        -- similarity or similarities need to be implemented/overridden
    """

    def __init__(self):
        self.SAME = 1
        self.DISTINCT = 0

    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        """
        similarity functions between two strings: sentence_1 and sentence_2
        returns a value between -1/0 and 1, where
            1 means same
            -1/0 means most distinct
        ==> bigger similarity value means higher similarity

        :param sentence_1:
        :param sentence_2:
        :return:
        """
        if sentence_1 == sentence_2:
            return self.SAME
        else:
            return self.DISTINCT

    def similarities(self, sentences_1: List[str], sentences_2: List[str]) -> List[float]:
        return [self.similarity(sentences_1[i], sentences_2[i]) for i in range(len(sentences_1))]


class TunedSentenceBertSimilarity(Similarity):
    def __init__(self, model_path='../../../models/02_small_reddit/30.000/sbert/'):
        super().__init__()
        self.sbert_model = None
        self.sbert_path = model_path

    def similarity(self, sentence_1: str, sentence_2: str) -> float:
        return self.similarities([sentence_1], [sentence_2])[0]

    def similarities(self, sentences_1: List[str], sentences_2: List[str], batch_size=EVAL_BATCH_SIZE) -> List[float]:
        if not self.sbert_model:
            logging.info("Loading model from {} ... ".format(self.sbert_path))
            self.sbert_model = SentenceTransformer(self.sbert_path)
            if self.sbert_path == 'roberta-base':
                # Roberta seems to be initialized with max seq length of 514 leading to errors in encoding
                self.sbert_model.max_seq_length = 512
            logging.info("Finished loading model ...")
        from style_similarity import cosine_sim
        with torch.no_grad():
            self.sbert_model.eval()
            return [cosine_sim(self.sbert_model.encode(u1, show_progress_bar=False, batch_size=batch_size),
                               self.sbert_model.encode(u2, show_progress_bar=False, batch_size=batch_size))
                    for u1, u2 in zip(sentences_1, sentences_2)]
