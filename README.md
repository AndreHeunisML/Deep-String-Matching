# Text Matching using Siamese Char CNN in Pytorch

## Context

Inspired by these papers

* https://arxiv.org/abs/1509.01626
* https://arxiv.org/abs/1702.02640

this is a system that takes a set of input characters (eg an employee name, a bank transaction text) and returns
the most likely match from a set of options (eg a database of employees / company names).  


## Structure

The system is trained in a siamese fashion, trying to minimise the triplet loss between matching strings. 
The idea is to first pretrain a model on a large set of different but related strings (misspelled words, 
lemmatisation, etc) and then adapt it for use on whatever data is available for the use case.

It is currently set up for use with lower-case letters and spaces. Sequences of numbers are replaced by a single 0.