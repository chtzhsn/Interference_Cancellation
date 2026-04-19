# # train a miniature character-level shakespeare model
# # good for debugging and playing on macbooks and such

# out_dir = 'out-shakespeare-char'
# eval_interval = 250 # keep frequent because we'll overfit
# eval_iters = 200
# log_interval = 10 # don't print too too often

# # we expect to overfit on this small dataset, so only save when val improves
# always_save_checkpoint = False

# wandb_log = False # override via command line if you like
# wandb_project = 'shakespeare-char'
# wandb_run_name = 'mini-gpt'

# dataset = 'shakespeare_char'
# gradient_accumulation_steps = 1
# batch_size = 64
# block_size = 256 # context of up to 256 previous characters

# # baby GPT model :)
# n_layer = 6
# n_head = 6
# n_embd = 384
# dropout = 0.2

# learning_rate = 1e-3 # with baby networks can afford to go a bit higher
# max_iters = 5000
# lr_decay_iters = 5000 # make equal to max_iters usually
# min_lr = 1e-4 # learning_rate / 10 usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# warmup_iters = 100 # not super necessary potentially

# # on macbook also add
# # device = 'cpu'  # run on cpu only
# # compile = False # do not torch compile the model


out_dir = 'out-shakespeare-char'
eval_interval = 50
eval_iters = 20
log_interval = 10

always_save_checkpoint = False
wandb_log = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 64

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.2
bias = False

learning_rate = 1e-3
max_iters = 300
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 20
lr_decay_iters = 300
min_lr = 1e-4

device = 'cpu'
compile = False

# interference cancellation
use_ic = False
ic_hidden_dim = 0
ic_lambda_base = 0.5
ic_lambda_mit = 1.0
