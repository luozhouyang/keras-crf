import unittest

import tensorflow as tf
from keras_crf.crf import CRF, CRFCategoricalAccuracy, CRFLoss


class CRFTest(unittest.TestCase):

    def test_crf(self):
        sequence_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='sequence_input')
        sequence_mask = tf.keras.layers.Lambda(lambda x: tf.greater(x, 0))(sequence_input)
        outputs = tf.keras.layers.Embedding(100, 128)(sequence_input)
        outputs = tf.keras.layers.Dense(256)(outputs)
        crf = CRF(7)
        # mask is important to compute sequence length in CRF
        outputs = crf(outputs, mask=sequence_mask)
        model = tf.keras.Model(inputs=sequence_input, outputs=outputs)
        model.compile(
            loss=CRFLoss(crf),
            metrics=[CRFCategoricalAccuracy(crf)],
            optimizer=tf.keras.optimizers.Adam(5e-5)
        )
        model.summary()


if __name__ == "__main__":
    unittest.main()
