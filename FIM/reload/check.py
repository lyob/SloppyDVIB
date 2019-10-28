import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file(file_name='./data-model/iterated-beta0-1500', tensor_name='', all_tensors=False)