import sys
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import AutoConfig, BertForSequenceClassificationWrapper

import crypten
import crypten.communicator as comm
from crypten.config import cfg
from utils import encrypt_tensor, encrypt_model

from models import Bert, BertEmbeddings

# Inference arguments
class config():
   def __init__(self):
       self.batch_size = 1
       self.num_hidden_layers = 12
       self.hidden_size = 768
       self.intermediate_size = 3072
       self.sequence_length = 8
       self.max_position_embeddings = 512
       self.hidden_act = "relu"
       self.softmax_act = "softmax"
       self.layer_norm_eps = 1e-12
       self.num_attention_heads = 12
       self.vocab_size = 28996
       self.hidden_dropout_prob = 0.1
       self.attention_probs_dropout_prob = 0.1

config = config()
print(f"using model config: {config}")

encrypt_weights = bool(int(sys.argv[1]))  # 0 is unencrypted, 1 is encrypted
crypten.init()

# setup fake data for timing purpose
#commInit = crypten.communicator.get().get_communication_stats()
input_ids = F.one_hot(torch.randint(
    low=0, high=config.vocab_size, size=(config.batch_size, config.sequence_length)), config.vocab_size).float()
x_enc = crypten.cryptensor(input_ids)

timing = defaultdict(float)

if encrypt_weights:
    m = Bert(config, timing)
    model = encrypt_model(m, Bert, (config, timing), input_ids).eval()
else:
    m = Bert(config, timing).eval()
    model = m

for i in range(10):
    #timing = defaultdict(float)
    model.reset_timing()
    time_s = time.time()
    # run a forward pass
    with crypten.no_grad():
        model(input_ids)
    torch.cuda.synchronize()

    time_e = time.time()
    timing["total_time"] = (time_e - time_s)
    print(timing)