import tensorflow as tf
import tensorflow_addons as tfa


def _unpack_data(data):
    """Support sample_weight"""
    if len(data) == 3:
        x, y, sample_weight = data
    else:
        x, y, sample_weight = data[0], data[1], None
    return x, y, sample_weight


class CRFModel(tf.keras.Model):

    def __init__(self,
                 model,
                 units: int,
                 chain_initializer="orthogonal",
                 use_boundary: bool = True,
                 boundary_initializer="zeros",
                 use_kernel: bool = True,
                 **kwargs):
        super().__init__()
        self.base_model = model
        self.units = units
        self.crf = tfa.layers.CRF(
            units=units,
            chain_initializer=chain_initializer,
            use_boundary=use_boundary,
            boundary_initializer=boundary_initializer,
            use_kernel=use_kernel,
            **kwargs)
        self.sequence_length = None

    def call(self, inputs, training=None, mask=None):
        """CRF endpoint forward pass.
        
        Args:
            inputs: Tensor, shape (batch_size, seq_len, hidden_size)

        Returns:
            potentials: Tensor, shape (batch_size, seq_len, units), in train phase
            sequence: Tensor, shape (batch_size, seq_len, units), in predict phase
        """
        outputs = self.base_model(inputs)
        sequence, potentials, self.sequence_length, _ = self.crf(outputs)
        if training:
            return potentials
        return tf.cast(tf.one_hot(sequence, self.units), dtype=self.dtype)

    def _compute_loss(self, y_true, y_pred, sample_weight=None):
        crf_loss = -tfa.text.crf_log_likelihood(y_pred, y_true, self.sequence_length, self.crf.chain_kernel)[0]
        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight
        crf_loss = tf.reduce_mean(crf_loss)
        return crf_loss

    def train_step(self, data):
        x, y, sample_weight = _unpack_data(data)
        with tf.GradientTape() as tape:
            # prediction is potentials
            prediction = self(x, training=True)
            crf_loss = self._compute_loss(y, prediction, sample_weight=sample_weight)
            total_loss = crf_loss + sum(self.losses)
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        y_pred = tf.argmax(prediction, axis=-1)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        results = {'loss': total_loss, 'crf_loss': crf_loss}
        results.update({m.name: m.result() for m in self.metrics})
        return results

    def test_step(self, data):
        x, y, sample_weight = _unpack_data(data)
        # prediction is viterbi decoded sequence
        prediction = self(x, training=False)
        crf_loss = self._compute_loss(y, prediction, sample_weight=sample_weight)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, tf.argmax(prediction, axis=-1))
        results = {'loss': crf_loss + sum(self.losses), 'crf_loss': crf_loss}
        results.update({m.name: m.result() for m in self.metrics})
        return results
