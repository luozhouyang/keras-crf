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


class ChinaPeopleDailyBuilder:

    def __init__(self, vocab_file, label_map=CHINA_PEOPLE_DAILY_LABELS, **kwargs):
        self.token_encoder = TokenMapper(vocab_file)
        self.label_encoder = LabelMapper(label_map)
        self.pad_id = self.token_encoder.pad_id

    def build_train_dataset(self, input_files, batch_size=32, buffer_size=30000, **kwargs):
        features, labels = read_conll_format_file(input_files)
        features = [self.token_encoder.encode(feature) for feature in features]
        labels = [self.label_encoder.encode(label) for label in labels]
        features = tf.ragged.constant(features, dtype=tf.int32)
        labels = tf.ragged.constant(labels, dtype=tf.int32)
        x_dataset = tf.data.Dataset.from_tensor_slices(features)
        x_dataset = x_dataset.map(lambda x: x)
        y_dataset = tf.data.Dataset.from_tensor_slices(labels)
        y_dataset = y_dataset.map(lambda y: y)
        dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
        dataset = dataset.filter(lambda x, y: tf.logical_and(tf.size(x) < 512, tf.size(y) < 512))
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=([None], [None]),
            padding_values=(self.pad_id, 0),
        )
        return dataset

    def build_valid_dataset(self, input_files, batch_size=32, buffer_size=5000, **kwargs):
        return self.build_train_dataset(input_files, batch_size=batch_size, buffer_size=buffer_size, **kwargs)

    def build_test_dataset(self, input_files, batch_size=32, **kwargs):
        features, labels = read_conll_format_file(input_files)
        features = [self.token_encoder.encode(feature) for feature in features]
        features = tf.ragged.constant(features)
        dataset = tf.data.Dataset.from_tensor_slices(features)
        dataset = dataset.filter(lambda x: tf.size(x) < 512)
        dataset = dataset.map(lambda x: x)  # convert to ragged tensor to normal tensor
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=[None],
            padding_values=self.pad_id,
        )
        dataset = dataset.map(lambda x: (x, None))
        return dataset
