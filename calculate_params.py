vocab = 50257
dmodel = 256
n_layers = 4

embedding = dmodel * vocab
head = dmodel * vocab
block = 12 * (dmodel ** 2)
total_params = embedding + n_layers * block + head

print(f'total_params: {total_params}')

n_tokens = total_params * 20

print(f'n_tokens: {n_tokens}')

batch_size = 256
seq_len = 256

n_steps = n_tokens // (batch_size * seq_len)

print(f'n_steps: {n_steps}')