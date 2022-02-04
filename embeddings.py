from emb_elmo_ci import context_insensitive_elmo
from emb_word2vec import read_word2vec


class EmbeddingsReader(object):
    def get_emb_w2v(self, options, embeddings_path, word2idx):
        embeddings, word2idx = read_word2vec(embeddings_path, word2idx)
        return embeddings, word2idx

    def get_emb_elmo(self, options, embeddings_path, word2idx):
        options_path = options.elmo_options_path
        weights_path = options.elmo_weights_path
        embeddings = context_insensitive_elmo(weights_path=weights_path, options_path=options_path,
            word2idx=word2idx, cuda=options.cuda, cache_dir=options.emb_cache_dir)
        return embeddings, word2idx

    def get_embeddings(self, options, embeddings_path, word2idx):
        if options.emb == 'w2v':
            out = self.get_emb_word2vec(options, embeddings_path, word2idx)
        elif options.emb == 'elmo':
            out = self.get_emb_elmo(options, embeddings_path, word2idx)
        return out
