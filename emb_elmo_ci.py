import hashlib
import os

import numpy as np
from tqdm import tqdm

from experiment_logger import get_logger
from pretrained_embeddings import get_character_embeddings_from_elmo, read_embeddings, write_embeddings, get_embeddings_path


def context_insensitive_elmo(weights_path, options_path, word2idx, cuda=False, cache_dir=None):
    """
    Embeddings are always saved in sorted order (by vocab) and loaded according to word2idx.
    """
    logger = get_logger()

    # Get tokens.
    idx2word = {i: w for w, i in word2idx.items()}
    tokens = [idx2word[i] for i in range(len(word2idx))]

    # get_character_embeddings_from_elmo

    if cache_dir is not None:
        cache_path = get_embeddings_path(tokens, cache_dir)

        logger.info('embedding cache path = {}'.format(cache_path))

        if os.path.exists(cache_path):
            logger.info('Loading embedding vectors: {}'.format(cache_path))
            vectors = read_embeddings(tokens, path=cache_path)
            logger.info('Embeddings with shape = {}'.format(vectors.shape))
            return vectors

    vectors = get_character_embeddings_from_elmo(tokens, cache_dir, cuda=cuda)

    if cache_dir is not None:
        logger.info('Saving embedding vectors: {}'.format(cache_path))
        write_embeddings(cache_path, vectors)

    logger.info('Embeddings with shape = {}'.format(vectors.shape))

    return vectors
