# -*- coding: utf-8 -*-
"""ONNX.ipynb

## Open Neural Network Exchange [ONNX]
"""

# !pip install onnx

# !pip install tensorflow-addons
# !git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow && pip install -e .

# !pip install torchvision

# !pip install onnx-tf

import torch
from models import customClassifier

import onnx
from onnx_tf.backend import prepare

model = customClassifier()

torch.save(model.state_dict(), 'jo.pth')

"""## Load the saved Pytorch model and export it as an ONNX file"""

model = customClassifier()
model.load_state_dict(torch.load('jo.pth'))

dummy_input = torch.randn(1,3,224,224)
dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(model, dummy_input, 'jo.onnx')

"""##Load the ONNX file and import it into Tensorflow"""

model_onnx = onnx.load("jo.onnx")
tf_ref = prepare(model_onnx) # tensorflow model

# saving tensorflow model
tf_ref.export_graph("jo.pb")