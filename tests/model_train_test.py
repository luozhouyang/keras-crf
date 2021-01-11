import unittest

import tensorflow as tf
import tensorflow_addons as tfa
from keras_crf.crf import CRF, CRFAccuracy, CRFLoss
from keras_crf.dataset import ChinaPeopleDailyBuilder, LabelMapper, TokenMapper

token_mapper = TokenMapper('data/vocab.txt')
label_mapper = LabelMapper()


class ModelTrainTest(unittest.TestCase):

    def _build_inputs(self):
        builder = ChinaPeopleDailyBuilder('data/vocab.txt')
        train_dataset = builder.build_valid_dataset('data/china-people-daily-ner-corpus/example.train')
        x = ['相', '比', '之', '下', '，', '青', '岛', '海', '牛', '队', '和', '广', '州', '松',
             '木', '队', '的', '雨', '中', '之', '战', '虽', '然', '也', '是', '0', ':', '0']
        tokens = token_mapper.encode(x)
        predict_inputs = tf.constant(tokens, shape=(1, len(tokens)), dtype=tf.int32)
        return train_dataset, predict_inputs

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
        dataset, predict_dataset = self._build_inputs()
        model.fit(dataset, steps_per_epoch=200)

        preds = model.predict(predict_dataset)
        for p in preds:
            print(p)
            print(label_mapper.decode(p))


if __name__ == "__main__":
    unittest.main()
