# PyTorch-Beam-Search
PyTorch implementation of beam search decoding for seq2seq models based on https://github.com/shawnwun/NNDIAL. 
Decoding goes seperately for each sentence and stores the nodes in prioritized queue.

Usage:
You can specify additional reward for decoding through BeamSearchNode.eval. Works for model with and without attention.
