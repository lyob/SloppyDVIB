# Sloppy models and Deep Variational Information Bottleneck exploration

Using Python version 3 and Tensorflow version 1.13

Based on the DVIB [paper](https://openreview.net/forum?id=HyxQzBceg) and [code](https://github.com/alexalemi/vib_demo) by Alemi, Murphy et al.

---
I recommend running the Python file on Visual Studio Code; this will allow each "cell" of code (demarcated by #%%) to be run individually, much like in a Jupyter notebook.

Code for data viewing/extraction and comment lines beginning with double hashtags (##) are by Ben Lyo.

---
### Explanation of files

*mnistvib_blyo_edit.py* is the code. A supplementary requirements file will be added soon.

The *DATA* folder contains checkpoint files generated when training the model. Each checkpoint consists of three files: a .meta file that describes the model graph, a .data file that holds the tensor values associated with variables, and an index file that indexes.

The *log_dir* folder is where any user outputs are stored to. *tensorweights.json* has all tensor weight and bias values.

Lastly, the *print_output* folder are a collection of manually created text files containing *print* outputs.