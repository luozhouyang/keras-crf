import unittest

import tensorflow as tf
from keras_crf.crf_model import CRFModel


class CRFModelTest(unittest.TestCase):

    def _build_model(self):
        sequence_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='sequence_input')
        outputs = tf.keras.layers.Embedding(21128, 128)(sequence_input)
        outputs = tf.keras.layers.Dense(256)(outputs)
        model = tf.keras.Model(inputs=sequence_input, outputs=outputs)
        crf_model = CRFModel(model, 5)
        crf_model.build(tf.TensorShape([None, 256]))
        return crf_model

    def test_build_model(self):
        model = self._build_model()
        model.compile(optimizer='adam', metrics=['acc'])
        model.summary()

    def test_train_model(self):
        model = self._build_model()
        model.compile(optimizer='adam', metrics=['acc'])


if __name__ == "__main__":
    unittest.main()
