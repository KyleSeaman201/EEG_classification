"""
Transformer model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np

import torch
from torch import nn
import random

####### Do not modify these imports.

class TransformerClassifier(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of features and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N, number of output
    classifications C.
    """
    def __init__(self,
                 input_size,
                 output_size,
                 device,
                 hidden_dim=128,
                 num_heads=2,
                 dim_feedforward=2048,
                 dim_k=96, dim_v=96, dim_q=96,
                 num_layers=1,
                 max_length=0):
        """
        :param input_size: the size of the input, which equals to the number of input values (ie sensors in EEG device)
        :param output_size: the size of the output, which equals to the number of classifications
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerClassifier, self).__init__()
        print("TransformerTranslator.hidden_dim:", hidden_dim)
        print("TransformerTranslator.num_heads:", num_heads)
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        assert(self.max_length != 0)
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q

        print("TransformerTranslator.device:", self.device)
        print("TransformerTranslator.input_size:", self.input_size)
        print("TransformerTranslator.output_size:", self.output_size)
        print("TransformerTranslator.hidden_dim:", self.hidden_dim)
        print("TransformerTranslator.dim_feedforward:", self.dim_feedforward)
        print("TransformerTranslator.dim_k:", self.dim_k)
        print("TransformerTranslator.dim_v:", self.dim_v)
        print("TransformerTranslator.dim_q:", self.dim_q)
                
        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        #self.embeddingL = nn.Embedding(self.input_size, self.hidden_dim)        #initialize word embedding layer
        #self.posembeddingL = nn.Embedding(self.max_length, self.hidden_dim)     #initialize positional embedding layer

        self.embeddingL = nn.Linear(self.input_size, self.hidden_dim)

        self.pos_dropout = nn.Dropout(0.2)
        # Create a long enough P
        self.P = torch.zeros((1, self.max_length, self.hidden_dim))
        X = torch.arange(self.max_length, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, self.hidden_dim, 2, dtype=torch.float32) / self.hidden_dim)
        print("self.P.shape: ", self.P.shape)
        print("X.shape: ", X.shape)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)


        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        
        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################
        
        # # Head #1
        # self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        # self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        # self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # # Head #2
        # self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        # self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        # self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
                
        # self.softmax = nn.Softmax(dim=2)
        # self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        # self.norm_mh = nn.LayerNorm(self.hidden_dim)

        # this uses the pytorch built-in TransformerEncoder builder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.hidden_dim,
            nhead = self.num_heads,
            dim_feedforward = self.dim_feedforward,
            dropout = 0.5,
            batch_first = True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer = self.encoder_layer,
            num_layers = self.num_layers
        )


        
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        # 
        # Don't forget the layer normalization.                                      #
        ##############################################################################
        # NOT NEEDED if using nn.TransformerEncoder
        # self.feed_forward = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.dim_feedforward),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(self.dim_feedforward, self.hidden_dim),
        #     nn.Dropout(0.2),
        # )
        # self.norm_ff = nn.LayerNorm(self.hidden_dim)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        self.output_layer = nn.Linear(self.hidden_dim, self.output_size)
        self.norm_out = nn.LayerNorm(self.hidden_dim)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,V,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling TransformerTranslator class methods here.      #
        #############################################################################
        #outputs = None      #remove this line when you start implementing your code
        outputs = self.final_layer(
            self.feedforward_layer(
                self.multi_head_attention(
                    self.embed(inputs)
                )
            )
        )
         
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def embed(self, inputs):
        """
        The model has an input sensor size of V, works on
        sequences of length T, has an hidden dimension of H, uses word vectors
        also of dimension H, and operates on minibatches of size N.

        :param inputs: intTensor of shape (N,V,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        #############################################################################
        # TODO:
        # Deliverable 1: Implement the embedding lookup.                            #
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        # This will take a few lines.                                               #
        #############################################################################
      
        #print("inputs.shape: ", inputs.shape)
        #print("inputs: ", inputs)

        # pos = torch.arange(self.max_length, dtype=torch.long).to(inputs.device).unsqueeze(0).expand_as(inputs)
        # embeddings = pos + inputs
        # embeddings = self.embeddingL(embeddings.transpose(2,1))
        # #print("new embeddings.shape: ", embeddings.shape)

        # TODO: try better position encodings.
        #print("inputs.shape: ", inputs.shape)
        #print("self.P.shape: ", self.P.shape)
        tmp = self.P[:, :inputs.shape[2], :].to(inputs.device)
        #print("tmp.shape: ", tmp.shape)
        tmp2 = self.embeddingL(inputs.transpose(2,1))
        #print("tmp2.shape: ", tmp2.shape)
        embeddings = tmp2 + tmp
        embeddings = self.pos_dropout(embeddings)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        
        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################
        # # # Head #1
        # # self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        # # self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        # # self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        # q1_x = self.q1(inputs)
        # k1_x = self.k1(inputs)
        # v1_x = self.v1(inputs)
        # #print("k1_x.shape: ", k1_x.shape)
        # score_1 = self.softmax(torch.matmul(q1_x, k1_x.transpose(1, 2)) / np.sqrt(self.dim_k))
        # score_1 = torch.matmul(score_1, v1_x)

        # # # Head #2
        # # self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        # # self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        # # self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
        # q2_x = self.q2(inputs)
        # k2_x = self.k2(inputs)
        # v2_x = self.v2(inputs)
        # score_2 = self.softmax(torch.matmul(q2_x, k2_x.transpose(1, 2)) / np.sqrt(self.dim_k))
        # score_2 = torch.matmul(score_2, v2_x)
        
        # #print("score_1.shape: ", score_1.shape)
        # #print("score_2.shape: ", score_2.shape)
        # z = self.attention_head_projection(torch.cat((score_1, score_2), 2))
        # #print("z.shape: ", z.shape)
        # outputs = self.norm_mh(inputs + z)

        outputs = self.encoder(inputs)

        #print("multi_head_attention: output.shape: ", outputs.shape)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################

        # not needed if using pytorch.TransformerEncoder
        # #print("inputs.shape: ", inputs.shape)
        # outputs = self.feed_forward(inputs)
        # #print("outputs2.shape: ", outputs.shape)
        # outputs = self.norm_ff(inputs + outputs)

        #print("feedforward_layer: output.shape: ", outputs.shape)

        outputs = inputs
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,C)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code. Softmax is not needed here    #
        # as it is integrated as part of cross entropy loss function.               #
        #############################################################################

        # TODO: we are given (N,T,C)... giving an output for every entry in the input
        # sequence (T), but we just want a final classifier probability.
        # What's the best way to summarize the "T" dimension?

        # this takes the mean of all in the output sequence
        outputs = torch.mean(self.norm_out(inputs), dim=1)

        # this takes the first of the output sequence, but doesn't seem to get as
        # good of results in training as the mean.
        #outputs = self.norm_out(inputs)[:, 0, :]

        outputs = self.output_layer(outputs)

        #print("final_layer: output.shape: ", outputs.shape)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
