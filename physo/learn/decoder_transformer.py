import torch
import numpy as np

import os
import psutil

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.d_k = embedding_dim // num_heads

        self.W_q = torch.nn.Linear(embedding_dim, embedding_dim)
        self.W_k = torch.nn.Linear(embedding_dim, embedding_dim)
        self.W_v = torch.nn.Linear(embedding_dim, embedding_dim)
        self.W_o = torch.nn.Linear(embedding_dim, embedding_dim)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q,K,V: (B, num_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, h, seq_q, seq_k)

        if mask is not None:
            # mask: (seq_len, seq_len) with True allowed -- expand to (1,1,seq_q,seq_k)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,seq,seq)
            # ensure mask dtype and device match scores
            mask = mask.to(dtype=torch.bool, device=scores.device)
            # replace disallowed (mask==False) positions with a large negative number
            scores = scores.masked_fill(~mask, -1e9)

        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, V)
        return output

    
    def split_input_into_heads(self, x):
        B, sequence_length, _ = x.shape
        return x.view(B, sequence_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_input_from_heads(self, x):
        B, _, sequence_length, d_k = x.shape
        return x.transpose(1,2).contiguous().view(B, sequence_length, self.embedding_dim)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_input_into_heads(self.W_q(Q))
        K = self.split_input_into_heads(self.W_k(K))
        V = self.split_input_into_heads(self.W_v(V))

        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_input_from_heads(attention_output))
        return output
    
class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, embedding_dim, ff_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(embedding_dim, ff_dim)
        self.fc2 = torch.nn.Linear(ff_dim, embedding_dim)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, embedding_dim, max_seq_length):
        super(PositionalEncoding, self).__init__()

        positional_encoding = torch.zeros(max_seq_length, embedding_dim)
        position_indices = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, embedding_dim, 2).float()*-np.log(10000.0)/embedding_dim)

        positional_encoding[:, 0::2] = torch.sin(position_indices*denominator)
        positional_encoding[:, 1::2] = torch.cos(position_indices*denominator)

        self.register_buffer('positional_encoding', positional_encoding.unsqueeze(1)) # (N, 1, embedding_dim)

    def forward(self, x):
        return x + self.positional_encoding[:x.shape[0]] # (seq_len, B, embedding_dim) + (seq_len, 1, embedding_dim)

class DecoderOnlyLayer(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout_fraction):
        super(DecoderOnlyLayer, self).__init__()
        self.attention_module = MultiHeadAttention(embedding_dim, num_heads)
        self.feed_forward = PositionWiseFeedForward(embedding_dim, ff_dim)
        self.ln1 = torch.nn.LayerNorm(embedding_dim)
        self.ln3 = torch.nn.LayerNorm(embedding_dim)
        self.dropout_module = torch.nn.Dropout(dropout_fraction)

    def forward(self, x, attention_mask):
        # encoder_out can be passed through two Linears giving different Q and K values
        attention_output = self.attention_module(x, x, x, attention_mask)
        x = self.ln1(x) + self.dropout_module(attention_output)
        ff_out = self.feed_forward(x)
        x = self.ln3(x) + self.dropout_module(ff_out)
        return x

class AttentionPool(torch.nn.Module):
    def __init__(self, max_sequence_length):
        super(AttentionPool, self).__init__()
        # self.sequence_weights = torch.nn.Parameter(torch.ones(1, max_sequence_length)/max_sequence_length)

    def forward(self, decoder_output):
        # sequence_length = decoder_output.shape[1]
        # decoder_output = self.sequence_weights[:, sequence_length] * decoder_output
        decoder_output = torch.mean(decoder_output, dim=1)
        return decoder_output

class Decoder(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout_fraction, input_size, output_size, is_lobotomized = False):
        super(Decoder, self).__init__()
        self.input_projection = torch.nn.Linear(input_size, embedding_dim)
        self.output_projection = torch.nn.Linear(embedding_dim, output_size)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_len)
        self.decoder = torch.nn.ModuleList([DecoderOnlyLayer(embedding_dim, num_heads, ff_dim, dropout_fraction) for _ in range(num_layers)])
        self.dropout_module = torch.nn.Dropout(dropout_fraction)
        self.previous_inputs = []
        self.is_lobotomized = is_lobotomized
        self.attention_pool = AttentionPool(max_sequence_length=max_seq_len)

    def generate_mask(self, target):
        # target: (seq_len, B, embedding_dim) or tensor with seq_len at dim 0
        sequence_length = target.shape[0]
        # lower triangular including diagonal -> True where allowed
        no_peek_mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=target.device))
        return no_peek_mask  # shape: (seq_len, seq_len) with True for allowed positions
    
    def reset(self):
        del self.previous_inputs
        self.previous_inputs = []
    
    def forward(self, input):
        self.previous_inputs.append(input.unsqueeze(0)) # [seq_len, B, input_dim]
        x_sequence = torch.cat(self.previous_inputs, dim=0) # (seq_len, B, input_dim)
        x_sequence = self.input_projection(x_sequence) # (seq_len, B, embedding_dim)
        causal_mask = self.generate_mask(x_sequence.detach()) # (seq_len, seq_len)
        x_sequence = self.dropout_module(self.positional_encoding(x_sequence)) # (seq_len, B, embedding_dim)
        x_sequence = x_sequence.transpose(0,1)
        for decoder_layer in self.decoder:
            x_sequence = decoder_layer(x_sequence, causal_mask)
        x_sequence = self.output_projection(x_sequence)
        x_sequence = self.attention_pool(x_sequence)
        return x_sequence
    
    def count_parameters (self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        return n_params
    
    def print_mem_cpu(self, label="", offset=0):
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024**2
        print(f"{label}: {(mem_mb-offset):.2f} MB")
        return mem_mb-offset
        



