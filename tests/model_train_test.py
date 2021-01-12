import os
import unittest

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras_crf.callbacks import EvaluateCallback
from keras_crf.crf import CRF, CRFAccuracy, CRFLoss
from keras_crf.dataset import ChinaPeopleDailyBuilder, LabelMapper, TokenMapper

token_mapper = TokenMapper('data/vocab.txt')
label_mapper = LabelMapper()

DATADIR = 'data/china-people-daily-ner-corpus'


class ModelTrainTest(unittest.TestCase):

    def _build_inputs(self):
        builder = ChinaPeopleDailyBuilder(token_mapper, label_mapper)
        train_dataset = builder.build_valid_dataset(os.path.join(DATADIR, 'example.train'))
        # x = ['相', '比', '之', '下', '，', '青', '岛', '海', '牛', '队', '和', '广', '州', '松',
        #      '木', '队', '的', '雨', '中', '之', '战', '虽', '然', '也', '是', '0', ':', '0']
        # tokens = token_mapper.encode(x)
        # predict_inputs = tf.constant(tokens, shape=(1, len(tokens)), dtype=tf.int32)
        predict_dataset, labels = builder.build_test_dataset(os.path.join(DATADIR, 'example.test'), with_labels=True)
        return train_dataset, predict_dataset, labels

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
            loss=CRFLoss(crf),
            metrics=[CRFAccuracy(crf)],
            optimizer=tf.keras.optimizers.Adam(4e-5)
        )
        return model

    def test_train_model(self):
        model = self._build_model()
        dataset, predict_dataset, labels = self._build_inputs()
        model.fit(
            dataset,
            steps_per_epoch=200,
            callbacks=[
                EvaluateCallback(token_mapper, label_mapper, os.path.join(DATADIR, 'example.dev')),
            ])

        preds = model.predict(predict_dataset, steps=1)
        for idx, p in enumerate(preds):
            # print(p)
            example = np.argmax(p, axis=-1).tolist()[:len(labels[idx])]
            print(labels[idx])
            print(label_mapper.decode(example))
            print()


if __name__ == "__main__":
    unittest.main()
