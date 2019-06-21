# Sloppy models and Deep Variational Information Bottleneck exploration

Using Python version 3 and Tensorflow version 1.13

Based on the DVIB [paper](https://openreview.net/forum?id=HyxQzBceg) and [code](https://github.com/alexalemi/vib_demo) by Alemi, Murphy et al.

---
I recommend running the Python file on Visual Studio Code; this will allow each code "cell" (demarcated by #%%) to be run individually, much like in a Jupyter notebook.

Code for data viewing/extraction and comment lines beginning with double hashtags (##) are by Ben Lyo.

To extract or view data (e.g. the weights and biases), skip or comment out the code cell that contains the training. 


---
### Explanation of files

*mnistvib_blyo_edit.py* is the code. A supplementary requirements file will be added soon.

The *DATA* folder contains checkpoint files generated when training the model. Each checkpoint consists of three files: an .index file that stores a list of variables names and shapes used, a .data file that holds the actual values of the variables saved, and a .meta file that contains all information required to restore a training or inference process.

The *log_dir* folder is where any user outputs are stored to. *tensorweights.json* (inside the zip file) has all tensor weight and bias values.

Lastly, the *print_output* folder are a collection of manually created text files containing *print* outputs.