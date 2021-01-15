import unittest

import tensorflow as tf
import tensorflow_addons as tfa
from keras_crf.crf import CRF


class ModelBuildTest(unittest.TestCase):

    def _build_model(self):
        sequence_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='sequence_input')
        sequence_mask = tf.keras.layers.Lambda(lambda x: tf.greater(x, 0))(sequence_input)
        outputs = tf.keras.layers.Embedding(21128, 128)(sequence_input)
        outputs = tf.keras.layers.Dense(256)(outputs)
        crf = CRF(7)
        # mask is important to compute sequence length in CRF
        outputs = crf(outputs, mask=sequence_mask)
        model = tf.keras.Model(inputs=sequence_input, outputs=outputs)
        model.compile(
            loss=crf.neg_log_likelihood,
            metrics=[crf.accuracy],
            optimizer=tf.keras.optimizers.Adam(4e-5)
        )
        return model

    def test_build_model(self):
        model = self._build_model()
        model.summary()


if __name__ == "__main__":
    unittest.main()
