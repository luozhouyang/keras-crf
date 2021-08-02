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
                 model: tf.keras.Model,
                 units: int,
                 chain_initializer="orthogonal",
                 use_boundary: bool = True,
                 boundary_initializer="zeros",
                 use_kernel: bool = True,
                 **kwargs):
        # build functional model
        crf = tfa.layers.CRF(
            units=units,
            chain_initializer=chain_initializer,
            use_boundary=use_boundary,
            boundary_initializer=boundary_initializer,
            use_kernel=use_kernel,
            **kwargs)
        # take model's first output passed to CRF layer
        decode_sequence, potentials, sequence_length, kernel = crf(inputs=model.outputs[0])
        # set name for outputs
        decode_sequence = tf.keras.layers.Lambda(lambda x: x, name='decode_sequence')(decode_sequence)
        potentials = tf.keras.layers.Lambda(lambda x: x, name='potentials')(potentials)
        sequence_length = tf.keras.layers.Lambda(lambda x: x, name='sequence_length')(sequence_length)
        kernel = tf.keras.layers.Lambda(lambda x: x, name='kernel')(kernel)
        super().__init__(
            inputs=model.inputs,
            outputs=[decode_sequence, potentials, sequence_length, kernel],
            **kwargs)
        self.crf = crf

    def train_step(self, data):
        x, y, sample_weight = _unpack_data(data)
        with tf.GradientTape() as tape:
            decode_sequence, potentials, sequence_length, kernel = self(x, training=True)
            crf_loss = -tfa.text.crf_log_likelihood(potentials, y, sequence_length, kernel)[0]
            if sample_weight is not None:
                crf_loss = crf_loss * sample_weight
            crf_loss = tf.reduce_mean(crf_loss)
            loss = crf_loss + sum(self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, potentials)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss})
        return results

    def test_step(self, data):
        x, y, sample_weight = _unpack_data(data)
        decode_sequence, potentials, sequence_length, kernel = self(x, training=False)
        crf_loss = -tfa.text.crf_log_likelihood(potentials, y, sequence_length, kernel)[0]
        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight
        crf_loss = tf.reduce_mean(crf_loss)
        loss = crf_loss + sum(self.losses)
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, potentials)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss})
        return results
