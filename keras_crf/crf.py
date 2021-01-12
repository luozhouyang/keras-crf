import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.metrics import MeanMetricWrapper
from tensorflow.python.keras.utils import losses_utils


class CRFLoss(LossFunctionWrapper):

    def __init__(self, crf, reduction=losses_utils.ReductionV2.AUTO, name='crf_loss', **kwargs):
        self.crf = crf
        fn = self._log_likelihood
        super().__init__(fn=fn, reduction=reduction, name=name, **kwargs)

    def _log_likelihood(self, y_true, y_pred):
        log_likelihood, _ = tfa.text.crf_log_likelihood(y_pred, y_true, self.crf.sequence_length, self.crf.chain_kernel)
        return tf.reduce_mean(-log_likelihood)


class CRFAccuracy(MeanMetricWrapper):

    def __init__(self, crf, name='crf_accuracy', dtype=None, **kwargs):
        self.crf = crf
        fn = self._accuracy
        super().__init__(fn=fn, name=name, dtype=dtype, **kwargs)

    def _accuracy(self, y_true, y_pred):
        if len(tf.keras.backend.int_shape(y_true)) == 3:
            y_true = tf.argmax(y_true, axis=-1)
        pred, _ = tfa.text.crf_decode(y_pred, self.crf.chain_kernel, self.crf.sequence_length)
        pred = tf.cast(pred, dtype=y_true.dtype)
        equals = tf.cast(tf.equal(y_true, pred), y_true.dtype)
        if self.crf.mask is not None:
            mask = tf.cast(self.crf.mask, y_true.dtype)
            equals = equals * mask
            return tf.reduce_sum(equals) / tf.reduce_sum(mask)
        return tf.reduce_mean(equals)


class CRF(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 chain_initializer='orthogonal',
                 use_boundary=True,
                 boundary_initializer='zeros',
                 use_kernel=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.crf = tfa.layers.CRF(
            units,
            chain_initializer=chain_initializer,
            use_boundary=use_boundary,
            boundary_initializer=boundary_initializer,
            use_kernel=use_kernel,
            **kwargs)
        self.num_class = units
        self.chain_kernel = self.crf.chain_kernel
        # record sequence length to compute loss
        self.sequence_length = None
        self.mask = None

    def call(self, inputs, training=None, mask=None):
        """Forward pass.

        Args:
            inputs: A [batch_size, max_seq_len, depth] tensor, inputs of CRF layer
            training: Boolean, in training process or not
            mask: A [batch_size, max_seq_len] boolean tensor, used to compulte sequence length in CRF layer

        Returns:
            potentials: A [batch_size, max_seq_len, num_class] tensor in train phase.
            sequence: A [batch_size, max_seq_len] tensor of decoded sequence in predict phase.
        """
        sequence, potentials, sequence_length, transitions = self.crf(inputs, mask=mask)
        # sequence_length is computed in both train and predict phase
        self.sequence_length = sequence_length
        # save mask, which is needed to compute accuracy
        self.mask = mask

        sequence = tf.cast(tf.one_hot(sequence, self.num_class), dtype=self.dtype)
        return tf.keras.backend.in_train_phase(potentials, sequence)
