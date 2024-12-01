import math

import torch
import torch.nn as nn
from rsl_rl.modules.rope import RotaryEmbedding

class MultiheadAttentionWithKVCache(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, context_length=128):
        super().__init__()
        self.embed_dim = embed_dim  # latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.context_length = context_length

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Projection layers for queries, keys, and values
        self.in_proj_q = nn.Linear(embed_dim, embed_dim)
        self.in_proj_k = nn.Linear(embed_dim, embed_dim)
        self.in_proj_v = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

        self.rotary_embedding = RotaryEmbedding(embed_dim // num_heads // 2, cache_if_possible=False, seq_before_head_dim=True)

    def forward(self, query, key, value, past_key=None, past_value=None, attn_mask=None, is_causal=False, use_cache=False, update_cache=False, position_step=-1):
        # query: (batch_size, query_len, embed_dim)
        # key, value: (batch_size, seq_len, embed_dim)
        # past_key, past_value: (batch_size, num_heads, past_seq_len, head_dim)
        batch_size = query.size(0)

        # Project inputs
        q = self.in_proj_q(query)  # (batch_size, query_len, embed_dim)
        k = self.in_proj_k(key)    # (batch_size, seq_len, embed_dim)
        v = self.in_proj_v(value)  # (batch_size, seq_len, embed_dim)

        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, query_len, head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

            
        if use_cache:
            seq_len = q.size(2)
            positions = position_step - seq_len + 1
            q = self.rotary_embedding.rotate_queries_or_keys(q, offset=positions, seq_dim=2)
            k = self.rotary_embedding.rotate_queries_or_keys(k, offset=positions, seq_dim=2)
        else:
            q = self.rotary_embedding.rotate_queries_or_keys(q, offset=0, seq_dim=2)
            k = self.rotary_embedding.rotate_queries_or_keys(k, offset=0, seq_dim=2)


        if past_key is not None and past_value is not None:
            # Concatenate past keys and values
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

            # Limit the cache to the context length
            k = k[:, :, -self.context_length:]
            v = v[:, :, -self.context_length:]

        else:
            # If no past keys/values, ensure current keys/values do not exceed context length
            k = k[:, :, -self.context_length:]
            v = v[:, :, -self.context_length:]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, query_len, key_len)

        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores += attn_mask  # attn_mask should be additive

        if is_causal:
            seq_len_q, seq_len_k = q.size(2), k.size(2)
            # TODO -- how to compute this without materialising this n^2 mask?
            causal_mask = torch.ones((seq_len_k, seq_len_k), device=q.device, dtype=torch.bool).tril(diagonal=0)[-seq_len_q:]
            attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0), float('-inf'))

        # Compute attention probabilities
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout_layer(attn_probs)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, v)  # (batch_size, num_heads, query_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # (batch_size, query_len, embed_dim)

        attn_output = self.out_proj(attn_output)  # (batch_size, query_len, embed_dim)

        # Return new keys and values for KV cache
        new_past_key = k if use_cache and update_cache else None
        new_past_value = v if use_cache and update_cache else None

        return attn_output, new_past_key, new_past_value

class Transformer_Block(nn.Module):
    def __init__(self, latent_dim, num_head, dropout_rate, context_length) -> None:
        super().__init__()
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.head_dim = latent_dim // num_head
        self.context_length = context_length
        self.ln_1 = nn.LayerNorm(latent_dim)
        self.attn = MultiheadAttentionWithKVCache(latent_dim, num_head, dropout=dropout_rate, context_length=context_length)
        self.ln_2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x, past_key=None, past_value=None, use_cache=False, update_cache=False, position_step=-1):
        # x: (batch_size, seq_len, latent_dim)
        # if no past kv, prepend x with zeros along seq_len to make context length
        # x: (batch_size, context_length, latent_dim)
        if past_key is None and past_value is None and use_cache and update_cache:
            prepend_size = max(self.context_length - x.size(1), 0)
            past_key = torch.zeros(x.size(0), self.num_head, prepend_size, self.head_dim, device=x.device)
            past_value = torch.zeros(x.size(0), self.num_head, prepend_size, self.head_dim, device=x.device)

        x_ln = self.ln_1(x)

        # Pass past_key and past_value to the attention module
        attn_output, new_past_key, new_past_value = self.attn(
            x_ln, x_ln, x_ln, past_key=past_key, past_value=past_value, is_causal=True, use_cache=use_cache, update_cache=update_cache, position_step=position_step
        )
        x = x + attn_output

        x = x + self.mlp(self.ln_2(x))

        return x, new_past_key, new_past_value

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, context_len, latent_dim=128, num_head=4, num_layer=4, dropout_rate=0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_len = context_len
        self.latent_dim = latent_dim
        self.num_head = num_head
        self.num_layer = num_layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )
        # Remove absolute positional embeddings
        # self.weight_pos_embed = nn.Embedding(context_len, latent_dim)
        self.layers = nn.ModuleList(
            [Transformer_Block(latent_dim, num_head, dropout_rate, context_len) for _ in range(num_layer)],
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, output_dim),
        )
        # Initialize the KV cache
        self.past_key_values = None  # Will be initialized per batch
        self.cache_enabled = True  # Flag to enable/disable cache

    def reset_cache(self, env_ids):
        # Reset the cache for specific environments
        if self.past_key_values is not None:
            for i in range(len(self.past_key_values)):
                past_key, past_value = self.past_key_values[i]
                if past_key is not None and past_value is not None:
                    # past_key, past_value: (batch_size, num_heads, seq_len, head_dim)
                    past_key[env_ids] = 0
                    past_value[env_ids] = 0

    def clear_cache(self):
        # Clear the entire cache
        self.past_key_values = None

    def enable_cache(self):
        self.cache_enabled = True

    def disable_cache(self):
        self.cache_enabled = False
        self.clear_cache()

    def forward(self, x, use_cache=False, update_cache=False, position_step=-1):
        # x: (batch_size, seq_len, input_dim)
        x = self.input_layer(x)

        new_past_key_values = []

        for i, layer in enumerate(self.layers):
            past_key, past_value = (None, None)
            if use_cache and self.cache_enabled and self.past_key_values is not None:
                past_key, past_value = self.past_key_values[i]

                if past_key is not None and past_value is not None:
                    seq_len = x.size(1)

                    end_pos = self.context_len - seq_len + 1

                    past_key = past_key[:, :, :end_pos, :]
                    past_value = past_value[:, :, :end_pos, :]

            x, new_k, new_v = layer(
                x, past_key=past_key, past_value=past_value, use_cache=use_cache, update_cache=update_cache, position_step=position_step
            )

            if use_cache and self.cache_enabled and update_cache:

                assert new_k.size(2) == new_v.size(2)
                assert new_k.size(2) <= self.context_len

                new_past_key_values.append((new_k, new_v))
            else:
                new_past_key_values.append((None, None))

        if use_cache and self.cache_enabled and update_cache:
            self.past_key_values = new_past_key_values

        x = self.output_layer(x)

        return x