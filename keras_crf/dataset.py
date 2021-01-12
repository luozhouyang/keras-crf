import abc

import tensorflow as tf

from . import utils


def read_conll_format_file(input_files, feature_index=0, label_index=1, **kwargs):
    all_features, all_labels = [], []

    def read_fn(features, labels):
        all_features.append(features)
        all_labels.append(labels)

    utils.read_conll_files(
        input_files,
        callback=read_fn,
        feature_index=feature_index,
        label_index=label_index,
        **kwargs)
    return all_features, all_labels


class TokenMapper:

    def __init__(self, vocab_file, unk_token='[UNK]', pad_token='[PAD]'):
        self.token2id = utils.load_vocab_file(vocab_file)
        self.id2token = {v: k for k, v in self.token2id.items()}
        assert len(self.token2id) == len(self.id2token)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.unk_id = self.token2id[unk_token]
        self.pad_id = self.token2id[pad_token]

    def encode(self, tokens, **kwargs):
        ids = [self.token2id.get(token, self.unk_id) for token in tokens]
        return ids

    def decode(self, ids, **kwargs):
        tokens = [self.id2token.get(_id, self.unk_token) for _id in ids]
        return tokens


CHINA_PEOPLE_DAILY_LABELS = {
    'O': 0,
    'B-LOC': 1,
    'I-LOC': 2,
    'B-ORG': 3,
    'I-ORG': 4,
    'B-PER': 5,
    'I-PER': 6
}


class LabelMapper:

    def __init__(self, label_map=CHINA_PEOPLE_DAILY_LABELS):
        self.label2id = label_map
        self.id2label = {v: k for k, v in self.label2id.items()}
        assert len(self.label2id) == len(self.id2label)

    def encode(self, labels, **kwargs):
        ids = [self.label2id.get(label, 0) for label in labels]
        return ids

    def decode(self, ids, **kwargs):
        labels = [self.id2label.get(_id, 'O') for _id in ids]
        return labels


class AbstractDatasetBuilder(abc.ABC):

    def build_train_dataset(self, input_files, **kwargs):
        raise NotImplementedError()

    def build_valid_dataset(self, input_files, **kwargs):
        raise NotImplementedError()

    def build_test_dataset(self, input_files, **kwargs):
        raise NotImplementedError()


class DatasetBuilder(AbstractDatasetBuilder):

    def __init__(self, token_mapper, label_mapper, token_pad_id=0, label_pad_id=0, **kwargs):
        super().__init__()
        self.token_mapper = token_mapper
        self.label_mapper = label_mapper
        self.token_pad_id = token_pad_id
        self.label_pad_id = label_pad_id

    def _read_files(self, input_files, **kwargs):
        raise NotImplementedError()

    def _build_dataset(self, features, labels, buffer_size=10000, sequence_maxlen=None, **kwargs):
        features = [self.token_mapper.encode(feature) for feature in features]
        labels = [self.label_mapper.encode(label) for label in labels]
        buffer_size = max(buffer_size, len(features))

        features = tf.ragged.constant(features, dtype=tf.int32)
        labels = tf.ragged.constant(labels, dtype=tf.int32)
        x_dataset = tf.data.Dataset.from_tensor_slices(features)
        x_dataset = x_dataset.map(lambda x: x)
        y_dataset = tf.data.Dataset.from_tensor_slices(labels)
        y_dataset = y_dataset.map(lambda y: y)
        dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
        # filter sequences
        if sequence_maxlen is not None and sequence_maxlen > 0:
            dataset = dataset.filter(lambda x, y: tf.logical_and(
                tf.size(x) < sequence_maxlen, tf.size(y) < sequence_maxlen))
        # shuffle dataset
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        return dataset

    def build_train_dataset(self, input_files, batch_size=32, buffer_size=10000, sequence_maxlen=None, **kwargs):
        features, labels = self._read_files(input_files, **kwargs)
        dataset = self._build_dataset(
            features,
            labels,
            buffer_size=buffer_size,
            sequence_maxlen=sequence_maxlen,
            **kwargs)
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=([None], [None]),
            padding_values=(self.token_pad_id, self.label_pad_id))
        return dataset

    def build_valid_dataset(self, input_files, batch_size=32, buffer_size=10000, sequence_maxlen=None, **kwargs):
        dataset = self.build_train_dataset(
            input_files=input_files,
            batch_size=batch_size,
            buffer_size=buffer_size,
            sequence_maxlen=sequence_maxlen,
            **kwargs)
        return dataset

    def _read_predict_files(self, input_files, **kwargs):
        raise NotImplementedError()

    def _build_predict_dataset(self, features, sequence_maxlen=None, **kwargs):
        features = [self.token_mapper.encode(feature) for feature in features]
        features = tf.ragged.constant(features)
        dataset = tf.data.Dataset.from_tensor_slices(features)
        if sequence_maxlen is not None and sequence_maxlen > 0:
            dataset = dataset.filter(lambda x: tf.size(x) < sequence_maxlen)
        dataset = dataset.map(lambda x: x)  # convert to ragged tensor to normal tensor
        return dataset

    def build_test_dataset(self, input_files, batch_size=32, sequence_maxlen=None, with_labels=True, **kwargs):
        outputs = self._read_predict_files(input_files, with_labels, **kwargs)
        if with_labels:
            features, labels = outputs
        else:
            features, labels = outputs, None
        dataset = self._build_predict_dataset(
            features=features,
            sequence_maxlen=sequence_maxlen,
            **kwargs)
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=[None],
            padding_values=self.token_pad_id)
        if with_labels:
            return dataset, labels
        return dataset


class ChinaPeopleDailyBuilder(DatasetBuilder):

    def _read_files(self, input_files, **kwargs):
        return read_conll_format_file(input_files, **kwargs)

    def _read_predict_files(self, input_files, with_labels=True, **kwargs):
        features, labels = read_conll_format_file(input_files, **kwargs)
        if with_labels:
            return features, labels
        return features
