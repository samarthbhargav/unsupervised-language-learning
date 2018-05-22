from api import SentEvalApi

import tensorflow as tf
from collections import defaultdict
import dill
import dgm4nlp


import numpy as np

class EmbeddingExtractor:
    """
    This will compute a forward pass with the inference model of EmbedAlign and
        give you the variational mean for each L1 word in the batch.

    Note that this takes monolingual L1 sentences only (at this point we have a traiend EmbedAlign model
        which dispenses with L2 sentences).

    You don't really want to touch anything in this class.
    """

    def __init__(self, graph_file, ckpt_path, config=None):
        g1 = tf.Graph()
        self.meta_graph = graph_file
        self.ckpt_path = ckpt_path

        self.softmax_approximation = 'botev-batch' #default
        with g1.as_default():
            self.sess = tf.Session(config=config, graph=g1)
            # load architecture computational graph
            self.new_saver = tf.train.import_meta_graph(self.meta_graph)
            # restore checkpoint
            self.new_saver.restore(self.sess, self.ckpt_path) #tf.train.latest_checkpoint(
            self.graph = g1  #tf.get_default_graph()
            # retrieve input variable
            self.x = self.graph.get_tensor_by_name("X:0")
            # retrieve training switch variable (True:trianing, False:Test)
            self.training_phase = self.graph.get_tensor_by_name("training_phase:0")
            #self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")

    def get_z_embedding_batch(self, x_batch):
        """
        :param x_batch: is np array of shape [batch_size, longest_sentence] containing the unique ids of words

        :returns: [batch_size, longest_sentence, z_dim]
        """
        # Retrieve embeddings from latent variable Z
        # we can sempale several n_samples, default 1
        try:
            z_mean = self.graph.get_tensor_by_name("z:0")

            feed_dict = {
                self.x: x_batch,
                self.training_phase: False,
                #self.keep_prob: 1.

            }
            z_rep_values = self.sess.run(z_mean, feed_dict=feed_dict)
        except:
            raise ValueError('tensor Z not in graph!')
        return z_rep_values

class EmbedAlignSentEval(SentEvalApi):

    def __init__(self, ckpt_path, tok_path, composition_method):
        super().__init__("embed_align")
        self.ckpt_path = ckpt_path
        self.tok_path = tok_path
        self.composition_method = composition_method

    def prepare(self, params, samples):
        """
        In this example we are going to load a tensorflow model,
        we open a dictionary with the indices of tokens and the computation graph
        """
        params.extractor = EmbeddingExtractor(
            graph_file='%s.meta'%(self.ckpt_path),
            ckpt_path=self.ckpt_path,
            config=None #run in cpu
        )

        # load tokenizer from training
        params.tks1 = dill.load(open(self.tok_path, 'rb'))
        params.composition_method = self.composition_method
        return

    def batcher(self, params, batch):
        """
        At this point batch is a python list containing sentences. Each sentence is a list of tokens (each token a string).
        The code below will take care of converting this to unique ids that EmbedAlign can understand.

        This function should return a single vector representation per sentence in the batch.
        In this example we use the average of word embeddings (as predicted by EmbedAlign) as a sentence representation.

        In this method you can do mini-batching or you can process sentences 1 at a time (batches of size 1).
        We choose to do it 1 sentence at a time to avoid having to deal with masking.

        This should not be too slow, and it also saves memory.
        """

        # if a sentence is empty dot is set to be the only token
        # you can change it into NULL dependening in your model
        batch = [sent if sent != [] else ['.'] for sent in batch]
        embeddings = []
        for sent in batch:
            # Here is where dgm4nlp converts strings to unique ids respecting the vocabulary
            # of the pre-trained EmbedAlign model
            # from tokens ot ids position 0 is en
            x1 = params.tks1[0].to_sequences([(' '.join(sent))])

            # extract word embeddings in context for a sentence
            # [1, sentence_length, z_dim]
            z_batch1 = params.extractor.get_z_embedding_batch(x_batch=x1)
            # sentence vector is the mean of word embeddings in context
            # [1, z_dim]
            sent_vec = np.mean(z_batch1, axis=1)
            # check if there is any NaN in vector (they appear sometimes when there's padding)
            if np.isnan(sent_vec.sum()):
                sent_vec = np.nan_to_num(sent_vec)
            embeddings.append(sent_vec)
        if len(embeddings) == 0:
            return None
        embeddings = np.vstack(embeddings)
        return embeddings
