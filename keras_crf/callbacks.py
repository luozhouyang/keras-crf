import numpy as np
import tensorflow as tf
from seqeval.metrics import classification_report

from .dataset import read_conll_format_file


class EvaluateCallback(tf.keras.callbacks.Callback):

    def __init__(self, token_mapper, label_mapper, valid_files, batch_size=32, sequence_maxlen=512):
        self.token_mapper = token_mapper
        self.laber_mapper = label_mapper
        self.batch_size = batch_size
        self.sequence_maxlen = sequence_maxlen

        self.valid_features, self.valid_labels = [], []
        features, labels = read_conll_format_file(valid_files)
        for f, l in zip(features, labels):
            if len(f) > sequence_maxlen:
                continue
            f = self.token_mapper.encode(f)
            # l = self.laber_mapper.encode(l)
            self.valid_features.append(f)
            self.valid_labels.append(l)

    def _build_dataset(self, features, batch_size=32):
        maxlen = max([len(x) for x in features])
        features = [x + [0] * (maxlen - len(x)) for x in features]
        x = tf.constant(features)
        x = tf.data.Dataset.from_tensor_slices(x)
        x = x.batch(batch_size)
        return x

    def on_epoch_end(self, epoch, logs=None):
        dataset = self._build_dataset(self.valid_features, batch_size=self.batch_size)
        # shape: [batch_size, max_seq_len, num_class]
        predictions = self.model.predict(dataset)
        # shape: [batch_size, max_seq_len]
        predictions = np.argmax(predictions, axis=-1)
        y_true = self.valid_labels
        y_pred = []
        for _pred, _true in zip(predictions, y_true):
            # remove paddings
            _pred = _pred[:len(_true)]
            tags = self.laber_mapper.decode(_pred.tolist())
            y_pred.append(tags)
        report = classification_report(y_true, y_pred)
        print()
        print('reports of epoch {}\n'.format(epoch + 1))
        print(report)
