# keras-crf

![Python package](https://github.com/luozhouyang/keras-crf/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/keras-crf.svg)](https://badge.fury.io/py/keras-crf)
[![Python](https://img.shields.io/pypi/pyversions/keras-crf.svg?style=plastic)](https://badge.fury.io/py/keras-crf)

A more elegant and convenient CRF built on tensorflow-addons.


> Python Compatibility is limited to [tensorflow/addons](https://github.com/tensorflow/addons), you can check the compatibility from it's home page.

## Installation

```bash
pip install keras-crf
```

## Usage

Here is an example to show you how to build a CRF model easily:

```python
import tensorflow as tf

from keras_crf import CRF, CRFLoss, CRFAccuracy


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
    metrics=[CRFAccuracy(crf)],
    optimizer=tf.keras.optimizers.Adam(5e-5)
    )
model.summary()
```

The model summary:

```bash
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
sequence_input (InputLayer)  [(None, None)]            0         
_________________________________________________________________
embedding (Embedding)        (None, None, 128)         12800     
_________________________________________________________________
dense (Dense)                (None, None, 256)         33024     
_________________________________________________________________
crf (CRF)                    (None, None)              1862      
=================================================================
Total params: 47,686
Trainable params: 47,686
Non-trainable params: 0
_________________________________________________________________
```
