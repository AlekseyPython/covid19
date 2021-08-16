import torch
from torch import nn


class PyTorchNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        nn.Module.__init__(self)
        
        #embedding layer
        self.embedding = nn.EmbeddingBag(num_embeddings = vocab_size, 
                                         embedding_dim  = embed_dim,
                                         mode           = 'sum')
        
        #lstm слой
        self.lstm = nn.LSTM(input_size = embed_dim, 
                           hidden_size = hidden_dim, 
                           num_layers  = 1, 
                           dropout     = dropout,
                           batch_first = True)
        
        # Полностью связанный слой
        self.fc = nn.Linear(in_features=hidden_dim, 
                            out_features=1)
        
        # Функция активации
        self.act = nn.Sigmoid()
        
        #init network small weights
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        
        self.lstm.bias_hh_l0.data.zero_()
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.weight_hh_l0.data.uniform_(-initrange, initrange)
        self.lstm.weight_hh_l0.data.uniform_(-initrange, initrange)
        
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, text, offsets):
        embedded       = self.embedding(text, offsets)
        embedded       = torch.reshape(embedded,   (embedded.shape[0], 1, embedded.shape[1]))
        
        _, (hidden, _) = self.lstm(embedded)
        dense_outputs  = self.fc(hidden.squeeze(0))
        outputs        = self.act(dense_outputs)
        outputs        = torch.reshape(outputs,   (outputs.shape[0],))
        return outputs
    
    
    