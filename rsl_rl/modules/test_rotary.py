import torch
from rope import RotaryEmbedding as NewRotaryEmbedding
from rope_old import RotaryEmbedding as OldRotaryEmbedding

# Set a fixed random seed for reproducibility
torch.manual_seed(42)

# Parameters
dim = 32
batch_size = 4
num_heads = 8
seq_len = 1024
head_dim = 64

# Scalar offset
offset_scalar = 200

# Tensor of offsets (different offset per batch element)
offsets_tensor = torch.randint(0, 100, (batch_size,))

# Instantiate both versions of RotaryEmbedding
rotary_emb_old = OldRotaryEmbedding(dim=dim)
rotary_emb_new = NewRotaryEmbedding(dim=dim)

# Generate mock queries and keys
q = torch.randn(batch_size, num_heads, seq_len, head_dim)
k = torch.randn(batch_size, num_heads, seq_len, head_dim)

### Test 1: Scalar offset ###

# Apply rotations using the old version (scalar offset)
q_old_scalar = rotary_emb_old.rotate_queries_or_keys(q, seq_dim=2, offset=offset_scalar)
k_old_scalar = rotary_emb_old.rotate_queries_or_keys(k, seq_dim=2, offset=offset_scalar)

# Apply rotations using the new version (scalar offset)
q_new_scalar = rotary_emb_new.rotate_queries_or_keys(q, seq_dim=2, offset=offset_scalar)
k_new_scalar = rotary_emb_new.rotate_queries_or_keys(k, seq_dim=2, offset=offset_scalar)

# Compare the outputs
q_diff_scalar = (q_old_scalar - q_new_scalar).abs().max()
k_diff_scalar = (k_old_scalar - k_new_scalar).abs().max()

print(f"Scalar offset test:")
print(f"Max difference in q: {q_diff_scalar.item()}")
print(f"Max difference in k: {k_diff_scalar.item()}")

if q_diff_scalar < 1e-6 and k_diff_scalar < 1e-6:
    print("Scalar offset test passed.")
else:
    print("Scalar offset test failed.")

### Test 2: Tensor offsets (multiple offsets in the same batch) ###

# Apply rotations using the old version (process each batch element separately)
q_old_list = []
k_old_list = []
for i in range(batch_size):
    q_i = q[i:i+1]  # shape (1, num_heads, seq_len, head_dim)
    k_i = k[i:i+1]
    offset_i = offsets_tensor[i].item()
    q_i_rotated = rotary_emb_old.rotate_queries_or_keys(q_i, seq_dim=2, offset=offset_i)
    k_i_rotated = rotary_emb_old.rotate_queries_or_keys(k_i, seq_dim=2, offset=offset_i)
    q_old_list.append(q_i_rotated)
    k_old_list.append(k_i_rotated)

# Concatenate the list to get the batch back
q_old_tensor = torch.cat(q_old_list, dim=0)
k_old_tensor = torch.cat(k_old_list, dim=0)

# Apply rotations using the new version with tensor offsets
q_new_tensor = rotary_emb_new.rotate_queries_or_keys(q, seq_dim=2, offset=offsets_tensor)
k_new_tensor = rotary_emb_new.rotate_queries_or_keys(k, seq_dim=2, offset=offsets_tensor)

# Compare the outputs
q_diff_tensor = (q_old_tensor - q_new_tensor).abs().max()
k_diff_tensor = (k_old_tensor - k_new_tensor).abs().max()

q_before_vs_after = (q_new_tensor - q).abs().max()
k_before_vs_after = (k_new_tensor - k).abs().max()

print(f"\nTensor offsets test:")
print(f"Offsets per batch element: {offsets_tensor.tolist()}")
print(f"Max difference in q: {q_diff_tensor.item()}")
print(f"Max difference in k: {k_diff_tensor.item()}")
# show that indeed they are different before and after
print(f"Max difference in q before vs after: {q_before_vs_after.item()}")
print(f"Max difference in k before vs after: {k_before_vs_after.item()}")

if q_diff_tensor < 1e-6 and k_diff_tensor < 1e-6:
    print("Tensor offsets test passed.")
else:
    print("Tensor offsets test failed.")
