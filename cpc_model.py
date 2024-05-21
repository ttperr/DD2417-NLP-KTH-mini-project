# First run this cell
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import math

class MultiHeadSelfAttention(nn.Module):
    """
    Calculates self-attention according to [Vaswani et al., NeurIPS, 2017]
    """
    def __init__(self, hidden_size, number_of_attention_heads):
        super().__init__()
        # The number of attention heads cannot be more than the hidden size
        assert hidden_size > number_of_attention_heads
        
        self.number_of_attention_heads = number_of_attention_heads
        # Divide the hidden_size roughly equal over the different heads
        self.attention_head_size = int(hidden_size / number_of_attention_heads)
        self.all_head_size = number_of_attention_heads * self.attention_head_size

        # Mapping from input to the query, key, and, value vectors
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=False)

        self.final = nn.Linear(self.all_head_size, hidden_size, bias=False)


    def reshape_for_multihead_attention(self, x):
        # x has the shape (batch_size, seq_length, hidden_size)
        B,S,_ = x.shape

        # but we want to split the representation of each token into 'number_of_heads' parts:
        x = x.reshape(B,S,self.number_of_attention_heads,self.attention_head_size)

        # and treat each part separately. Thus, we need the final tensor to have shape
        # (batch_size, number_of_heads, seq_length, attention_head_size)
        return x.permute(0, 2, 1, 3)

    
    def forward(self, hidden_states):
        # All of the tensors below will have the shape (batch_size, seq_length, hidden_size)
        query_all_heads = self.query( hidden_states )
        key_all_heads = self.key( hidden_states )  
        value_all_heads = self.value( hidden_states ) 

        # All of the tensors below will have the shape (batch_size, number_of_heads, seq_length, attention_head_size) 
        Q = self.reshape_for_multihead_attention( query_all_heads )
        K = self.reshape_for_multihead_attention( key_all_heads )
        V = self.reshape_for_multihead_attention( value_all_heads )

        # attention_scores will have the shape(batch_size, number_of_heads, seq_length, seq_length)
        attention_scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) 

        # Scale to reduce variance
        attention_scores /= torch.sqrt(torch.tensor(self.attention_head_size).float())

        # Use softmax to turn the attention scores into probabilities.
        # We want zero scores to be zero probabilities -- hence we turn
        # zero scores into -infinity before the softmax exponentiation.
        
        # YOUR CODE HERE
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, V)
        
        # Now produce the contextualized vectors for each head
        # The tensor below will have shape (batch_size, number_of_heads, seq_length, head_size)
        self_attention_all_heads_separately = torch.matmul(attention_probs, V)
        self_attention_all_heads_separately = attention_output.permute(0, 2, 1, 3).contiguous().view(hidden_states.size(0), -1, self.all_head_size)

        # For each token, we now want to bring together the representation coming from each head.
        # The 'self_attention' tensor below should have the shape 
        # (batch size, seq_length_, self.all_heads_size)
        self_attention = attention_output.permute(0, 2, 1, 3).contiguous().view(hidden_states.size(0), -1, self.all_head_size)

        # Finally, make sure that the output has the correct dimensions (batch_size,seq_length,hidden_size)
        return self.final( self_attention )
    
    
class PositionwiseFFN(nn.Module):
    """
    The position-wise FFN that follows after the self-attention
    computation.
    """
    def __init__(self, hidden_size, dropout_prob) :
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout_prob)
        for module in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0.)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class EncoderBlock(nn.Module):
    """
    Transformer encoder block.
    
    This version differs from the original version in  [Vaswani et al. NeurIPS 2017],
    and applies the LayerNorm before the self-attention, and before the FFN, as this
    has proved to be beneficial (see [Nguyen and Salazar 2019]).
    """
    def __init__(self, hidden_size, number_of_attention_heads, dropout_prob) :
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden_size, number_of_attention_heads)
        self.ffn = PositionwiseFFN(hidden_size, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        x1 = self.ln1(x)
        x2 = x + self.dropout(self.attn(x1))
        x3 = self.ln2(x2)
        x4 = x2 + self.dropout(self.ffn(x3))
        return x4


# ======================= The model ======================= #

class CharLM(nn.Module) :

    def __init__(self, config, no_of_input_chars, MAXLEN ) :
        super(CharLM, self).__init__()
        self.MAXLEN = MAXLEN
        self.config = config
        self.embed = nn.Embedding(no_of_input_chars,config.hidden_size)
        # Make sure that the padding symbol (which has ID 0) is embedded
        # as a vector of 0s.
        self.embed.weight.data[0].fill_(0)
        self.positional = nn.Parameter(torch.randn(1, self.MAXLEN, config.hidden_size))
        modules = [EncoderBlock(config.hidden_size, 
                                config.number_of_attention_heads,
                                config.dropout_prob) for _ in range(config.number_of_transformer_encoders)]
        self.transformers = nn.ModuleList(modules)
        self.final = nn.Linear(config.hidden_size*self.MAXLEN, no_of_input_chars)

    def forward(self,x) :
        number_of_datapoints = x.shape[0]
        # First create a mask distinguishing 0 from positive word IDs
        non_zero_mask = (x != 0)
        word_embeddings = self.embed(x)
        # Add positional vectors in all non-padded positions
        pos = self.positional.expand_as(word_embeddings)
        pos = pos * non_zero_mask.unsqueeze(-1).float()
        t = word_embeddings + pos
        # Then apply the transformers and make a final prediction at the end
        for transf in self.transformers :
            t = transf(t)
        flattened_transf = t.reshape(number_of_datapoints,1,-1)
        result =  self.final(torch.tanh(flattened_transf))
        return result