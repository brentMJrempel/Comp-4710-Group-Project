# Comp-4710-Group-Project
##A deep learning model to predict peptide binding from protein sequences


This project runs on Python 2.7 and uses Theano as a backend for Keras. The official Keras install process was used on Ubuntu 16.10. The only initial install and configuration gotchas encountered so far (if they could be called gotchas) were using the latest version of `pip`, and remembering to set `"backend": "theano"` in `keras.json`. Windows should work too. Currently it's tested as pretty fast on a CPU.

Right now `Dataset.txt` is read directly. It contains protein's amino acid sequences followed by their locations that bind peptides.
The files `Train.txt`, and `Test.txt`, contain references to individual residues used in the [Taherzadeh et al paper](https://scholar.google.ca/scholar?q=sequence+based+prediction+of+protein+peptide+binding+sites+using+support+vector+machine) for training and testing respectively.
