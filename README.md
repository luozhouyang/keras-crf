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

from keras_crf import CRFModel

# build backbone model, you can use large models like BERT
sequence_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='sequence_input')
outputs = tf.keras.layers.Embedding(21128, 128)(sequence_input)
outputs = tf.keras.layers.Dense(256)(outputs)
base = tf.keras.Model(inputs=sequence_input, outputs=outputs)

# build CRFModel, 5 is num of tags
model = CRFModel(base, 5)

# no need to specify a loss for CRFModel, model will compute crf loss by itself
model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4)
    metrics=['acc'],
    )
model.summary()

# you can now train this model
model.fit(dataset, epochs=10, callbacks=None)
```

The model summary:

```bash
Model: "crf_model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
sequence_input (InputLayer)     [(None, None)]       0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, None, 128)    2704384     sequence_input[0][0]             
__________________________________________________________________________________________________
dense (Dense)                   (None, None, 256)    33024       embedding[0][0]                  
__________________________________________________________________________________________________
crf (CRF)                       [(None, None), (None 1320        dense[0][0]                      
__________________________________________________________________________________________________
decode_sequence (Lambda)        (None, None)         0           crf[0][0]                        
__________________________________________________________________________________________________
potentials (Lambda)             (None, None, 5)      0           crf[0][1]                        
__________________________________________________________________________________________________
sequence_length (Lambda)        (None,)              0           crf[0][2]                        
__________________________________________________________________________________________________
kernel (Lambda)                 (5, 5)               0           crf[0][3]                        
==================================================================================================
Total params: 2,738,728
Trainable params: 2,738,728
Non-trainable params: 0
__________________________________________________________________________________________________
```
