import logging
import os


def read_files(input_files, callback=None, **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        if not os.path.exists(f):
            logging.warning('File %s does not exist.', f)
            continue
        logging.info('Starting to read file %s', f)
        with open(f, mode='rt', encoding=kwargs.get('encoding', 'utf-8')) as fin:
            for line in fin:
                line = line.rstrip('\n')
                if callback:
                    callback(line)
        logging.info('Finished to read file %s', f)


def read_conll_files(input_files, callback=None, **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    feature_index, label_index = kwargs.get('feature_index', 0), kwargs.get('label_index', 1)
    for f in input_files:
        if not os.path.exists(f):
            logging.warning('File %s does not exist.', f)
            continue
        logging.info('Starting to read file %s', f)
        with open(f, mode='rt', encoding=kwargs.get('encoding', 'utf-8')) as fin:
            lines = fin.read().splitlines()
        features, labels = [], []
        for line in lines:
            parts = line.split(' ')
            if len(parts) == 1:
                if callback:
                    callback(features, labels)
                features, labels = [], []
            else:
                features.append(parts[feature_index])
                labels.append(parts[label_index])
        logging.info('Finished to read file %s', f)


def load_vocab_file(vocab_file, **kwargs):
    vocabs = {}
    lino = 0
    with open(vocab_file, mode='rt', encoding='utf8') as fin:
        for line in fin:
            v = line.rstrip('\n')
            vocabs[v] = lino
            lino += 1
    return vocabs
