import torch
import torch.nn as nn
import math
from cuda.binding import strided_attention_forward # This will be our compiled CUDA kernel

def naive_strided_attention(q, k, v, stride):
    """
    Naive, pure-PyTorch implementation of strided attention.
    This is your reference for correctness.

    Args:
        q, k, v: Tensors of shape (batch_size, num_heads, seq_len, head_dim)
        stride: The stride for sparse attention.
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Create an empty output tensor
    output = torch.zeros_like(q)
    
    # Scale factor for dot products
    scale = 1.0 / math.sqrt(head_dim)

    # Iterate over each item in the batch and each head
    for b in range(batch_size):
        for h in range(num_heads):
            # Iterate over each query token
            for i in range(seq_len):
                q_vec = q[b, h, i, :]
                
                # --- Attention Score Calculation ---
                scores = torch.zeros(seq_len, device=q.device)
                # Only compute scores for strided keys
                for j in range(0, seq_len, stride):
                    k_vec = k[b, h, j, :]
                    scores[j] = torch.dot(q_vec, k_vec) * scale
                
                # --- Softmax ---
                # Mask out non-strided scores before softmax
                mask = torch.full((seq_len,), float('-inf'), device=q.device)
                mask[::stride] = 0
                scores += mask
                
                attn_probs = torch.softmax(scores, dim=-1)

                # --- Weighted Sum of Values ---
                out_vec = torch.zeros(head_dim, device=q.device)
                for j in range(0, seq_len, stride):
                    v_vec = v[b, h, j, :]
                    out_vec += attn_probs[j] * v_vec
                
                output[b, h, i, :] = out_vec
                
    return output


class CustomStridedAttention(nn.Module):
    """
    This module will use your custom CUDA kernel for strided attention.
    """
    def __init__(self, embed_dim, num_heads, stride):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.stride = stride
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape

        # Project and reshape Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # TODO: Replace the call to the naive implementation below with a call
        # to your custom CUDA kernel `strided_attention_forward`.
 
        # Using the naive implementation as a placeholder.
        output = naive_strided_attention(q, k, v, self.stride)

        # Reshape and project the output back to the original shape
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        return output
