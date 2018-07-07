import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from snorkel.learning.pytorch.rnn.rnn_base import RNNBase
from snorkel.learning.pytorch.rnn.utils import SymbolTable, candidate_to_tokens

######################################################################################################
##### CLASSES DEFINING DISCRIMINATIVE MODEL
######################################################################################################

class MemexRNNBase(RNNBase):
    def _preprocess_data(self, candidates, extend=False):
        """Convert candidate sentences to lookup sequences, subclass of Snorkel RNNBase
        
        :param candidates: candidates to process
        :param extend: extend symbol table for tokens (train), or lookup (test)?
        """
        if not hasattr(self, 'word_dict'):
            self.word_dict = SymbolTable()
        data, ends = [], []
        for candidate in candidates:
            s = candidate_to_tokens(candidate)
            # Either extend word table or retrieve from it
            f = self.word_dict.get if extend else self.word_dict.lookup
            data.append(np.array(list(map(f, s))))
            
        return data

class LSTM(MemexRNNBase):
    """
    Same as Snorkel LSTM, but based on MemexRNNBase to deal with extraction vs. RE use case
    """
    def _build_model(self, embedding_dim=50, hidden_dim=50, num_layers=1, dropout=0.25, bidirectional=False,
                     word_dict=SymbolTable(), **kwargs):
        self.word_dict = word_dict
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.word_dict.len(), self.embedding_dim, padding_idx=0)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim,
                            num_layers=num_layers, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True
                            )

        self.output_layer = nn.Linear(hidden_dim * self.num_directions, self.cardinality if self.cardinality > 2 else 1)
        self.dropout_layer = nn.Dropout(p=dropout)
        
    def forward(self, X, hidden_state):
        seq_lengths = torch.zeros((X.size(0)), dtype=torch.long)
        for i in range(X.size(0)):
            for j in range(X.size(1)):
                if X[i, j] == 0:
                    seq_lengths[i] = j
                    break
                seq_lengths[i] = X.size(1)

        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        X = X[perm_idx, :]
        inv_perm_idx = torch.tensor([i for i, _ in sorted(enumerate(perm_idx), key=lambda idx: idx[1])], dtype=torch.long)

        encoded_X = self.embedding(X)
        encoded_X = pack_padded_sequence(encoded_X, seq_lengths, batch_first=True)
        _, (ht, _) = self.lstm(encoded_X, hidden_state)
        output = ht[-1] if self.num_directions == 1 else torch.cat((ht[0], ht[1]), dim=1)

        return self.output_layer(self.dropout_layer(output[inv_perm_idx, :]))
    
    def initialize_hidden_state(self, batch_size):
        return (
            torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim),
            torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim)
        )