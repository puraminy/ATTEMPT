import wandb
import re
from pathlib import Path
import transformers
import numpy as np
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import os
import math
from os.path import expanduser
import attempt.mylogs as mylogs

from transformers import AddedToken 
from transformers.optimization import Adafactor, AdamW
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from attempt.maps import *
from collections import OrderedDict

def _isin(tensor:torch.Tensor,values:torch.Tensor):
    return (tensor[..., None] == values).any(-1)

class PromptEncoder(torch.nn.Module):
    enc_type = "encoder"
    def __init__(self, name, prompt_tokens, length=None, model=None, tokenizer=None): 
        super().__init__()
        self.name = name
        self.prompt_tokens = prompt_tokens
        self.length = len(prompt_tokens) if prompt_tokens else length
        self.embedding_dim = model.config.hidden_size
        self.embedding = torch.nn.Embedding(self.length, self.embedding_dim)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net_inps = torch.arange(self.length, device=self.device)

        self.prompt_ids = self.get_prompt_ids(prompt_tokens, model, tokenizer)
        self.input_ids = torch.tensor(self.prompt_ids, device=self.device)
        self.id_offset = min(self.prompt_ids) if self.prompt_ids else 0 
        self.is_source = False
        self.src_idx = -1
        self.attend_to = None

    def get_prompt_ids(self, prompt_tokens, model, tokenizer, init_emb_flag = True):
        prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)

        if not init_emb_flag:
            return prompt_ids 

        cur_embeddings = model.get_input_embeddings()
        init_embs = {}
        mylogs.bp("encoder")

        for pid, p in enumerate(prompt_ids):
            if pid < cur_embeddings.num_embeddings:
                emb = cur_embeddings.weight[pid,:].detach().clone() 
                init_embs[pid] = emb

        # init from words
        for pid, p in enumerate(prompt_tokens):
            if "_" in p:
               q = p.strip("<").strip(">")
               w = q.split("_")[1]
               if "?" in w: w = w.split("?")[0]
               if not w.isdigit():
                   wid = tokenizer.convert_tokens_to_ids([w])[0]
                   emb = cur_embeddings.weight[wid,:].detach().clone() 
                   init_embs[pid] = emb

        self.init_embs = init_embs
        self.init_embedding(init_embs)
        return prompt_ids 

    def get_filename(self, length=None, prefix="pt"):
        length = length if length is not None else self.length
        fname = prefix + "_" + self.enc_type + "_" + self.name + "_" + str(length) + ".pt"
        if self.is_source:
            fname = fname.replace("source_","") 
        return fname

    def save(self, save_dir, prefix="pt"):
        fname = os.path.join(save_dir, self.get_filename(prefix=prefix))
        state_dict = self.state_dict()
        torch.save(state_dict, fname)

    def load(self, load_dir, prefix="pt", length = None):
        fname = os.path.join(load_dir, self.get_filename(length, prefix))
        assert Path(fname).is_file(), fname + " doesn't exists to be loaded!"
        mapl=torch.device('cpu')

        state = torch.load(fname, map_location=mapl)
        size = state["embedding.weight"].size()

        self.length = size[0]
        self.embedding_dim = size[1] 
        self.embedding = torch.nn.Embedding(self.length, self.embedding_dim)
        self.load_state_dict(state)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net_inps = torch.arange(self.length, device=self.device)
        mylogs.tinfo("Prompt for %s was loaded ", self.name)

    def init_embedding(self, init_embs):
        if init_embs:
            with torch.no_grad():
                for _id,emb in init_embs.items():
                    if _id < len(self.embedding.weight):
                        self.embedding.weight[_id] = emb
        else:
            random_range = 0.5
            self.embedding.weight.data.uniform_(-random_range, random_range)

    def init_embs_from_ids(self, embeds):
        embs = {}
        for i, pid in enumerate(self.prompt_ids):
           if pid < len(embeds.weight):
               emb = embeds.weight[pid,:].detach().clone() 
               self.embedding.weight[i] = emb

    def init_embs_from_words(self, embeds):
        indices = np.random.permutation(range(5000))[:self.length]
        init_weight = embeds.state_dict()[
            "weight"][indices]
        self.embedding.weight.data = init_weight.clone().detach()

    def forward(self,prompt_token_ids, tids=None, training=True):
        if tids is not None:
            task_id = tids[0]
        index_list = prompt_token_ids
        if self.input_ids.size()[0] > 0:
            index_list = (prompt_token_ids.view(-1,1) == self.input_ids).int().argmax(dim=1)
        ret_embeds = self.forward_step(index_list, tids, training)
        return ret_embeds

    def forward_step(self, index_list, tids=None, training=True):
        raise NotImplementedError()

    def isin(self, ar1, ar2):
        return (ar1[..., None] == ar2).any(-1)

    def get_prompt_token_fn(self):
        if self.input_ids is not None:
            return lambda x: self.isin(x, self.input_ids)
        else:
            return lambda x: (x>=self.id_offset)&(x<self.id_offset+self.length)

    def dump_embeddings_into(self, weight, task_ids = None):
        if task_ids == None:
            task_ids = [0]
        with torch.no_grad():
            embs = self.forward(self.input_ids, tids=task_ids, training=False)
            detached_embeddings = embs.detach()
            weight[self.prompt_ids,:]=detached_embeddings

    def get_emb(self, task_ids = None):
        with torch.no_grad():
            embs = self.forward(self.input_ids, tids=task_ids, training=False)
            detached_embeddings = embs.detach()
            return detached_embeddings

class EmbeddingPromptEncoder(PromptEncoder):
    def forward_step(self, index_list, tids=None, training=True):
        ret_embeds = self.embedding(index_list)
        return ret_embeds 

class MatPromptEncoder(PromptEncoder):
    def __init__(self, prefix_config, **kwargs):
        super().__init__(**kwargs)
        self.prefix_config = prefix_config
        if prefix_config is not None:
            self.temperature = self.prefix_config['temperature']
            self.z = nn.Parameter(data=torch.empty((
                prefix_config['n_prompts'],
                prefix_config['intrinsic_dim']
            )).uniform_(-1e-3, 1e-3))

            bound = 1 / math.sqrt(prefix_config['n_prompt_tokens'] * self.embedding_dim)
            self.A = nn.Parameter(data=torch.empty((
                prefix_config['intrinsic_dim'],
                prefix_config['n_prompt_tokens'] * self.embedding_dim
            )).uniform_(-bound, bound), requires_grad=False)

    def forward_step(self, index_list, tids=None, training=True):
        running_weight = torch.mm(self.z, self.A) 
        ret_embeds = F.embedding(index_list, running_weight)
        return ret_embeds 

class MLPPromptEncoder(PromptEncoder):
    enc_type = "mlp"
    def __init__(self, num_layers=1, hidden_size=-1, **kwargs) -> None:
        super().__init__(**kwargs)
        embedding_dim = self.embedding_dim
        hsize = hidden_size if hidden_size > 1 else embedding_dim
        if num_layers == 2:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, hsize),
                torch.nn.ReLU(),
                torch.nn.Linear(hsize, hsize),
                torch.nn.ReLU(),
                torch.nn.Linear(hsize, embedding_dim)
            )
        else:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, hsize),
                torch.nn.ReLU(),
                torch.nn.Linear(hsize, embedding_dim)
            )

    def forward_step(self, index_list, tids=None, training=True):
        embs = self.embedding(self.net_inps)
        z = self.mlp(embs)
        ret_embeds = F.embedding(index_list, z)
        return ret_embeds 

class LSTMEmbeddingPromptEncoder(PromptEncoder):
    enc_type = "lstm"
    def __init__(self,num_layers=1, hidden_size=-1, **kwargs) -> None:
        mylogs.bp("encoder|lstm")
        super().__init__(**kwargs)
        embedding_dim = self.embedding_dim
        hsize = hidden_size if hidden_size > 1 else embedding_dim
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim // 2, #my code
            num_layers=2,
            dropout=0,
            bidirectional=True,
            batch_first=True
        )
        if num_layers == 2:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, embedding_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(embedding_dim, hsize),
                torch.nn.ReLU(),
                torch.nn.Linear(hsize, embedding_dim)
            )
        else:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, hsize),
                torch.nn.ReLU(),
                torch.nn.Linear(hsize, embedding_dim)
            )

 #### llllllf
    def forward_step(self, index_list, tids=None, training=True):
        net_inputs = self.net_inps
        # create embedding vectors for input ids
        embeds = self.embedding(net_inputs)
        x = self.lstm(embeds.unsqueeze(0))
        running_weight = self.mlp(x[0]).squeeze(0)
        ret_embeds = F.embedding(index_list, running_weight)
        return ret_embeds

def add_specials(tokenizer):
    cur_list = tokenizer.additional_special_tokens
    num_added_toks: dict = {}
    if tokenizer.bos_token is None:
        num_added_toks['bos_token'] = "<s>"
    if tokenizer.eos_token is None:
        num_added_toks['eos_token'] = "</s>"
    if tokenizer.pad_token is None:
        num_added_toks['pad_token'] = "<pad>"
    if tokenizer.sep_token is None:
        num_added_toks['sep_token'] = "<sep>"
    if tokenizer.cls_token is None:
        num_added_toks['cls_token'] = "<cls>"
    if tokenizer.mask_token is None:
        num_added_toks['mask_token'] = "<mask>"

    num_tokens = tokenizer.add_special_tokens(num_added_toks)
    new_tokens = list(set(REL_TO_TOKEN.values()))+ \
                 list(set(GEN_TOKENS.values())) 
    added_tokens = [ 
            AddedToken(tok,lstrip=True,
                rstrip=True, single_word=True)
            for tok in new_tokens if not tok in cur_list
    ]
    added_tokens = cur_list + added_tokens
    num_tokens += tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
    return num_tokens

def extend_tokenizer(tokenizer, prompt_tokens = []):
    cur_list = tokenizer.additional_special_tokens
    new_tokens = []
    new_tokens += prompt_tokens
    added_tokens = [ 
            AddedToken(tok,lstrip=True,
                rstrip=False, single_word=True)
            for tok in new_tokens if not tok in cur_list
    ]
    if added_tokens:
        added_tokens = cur_list + added_tokens
        tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})

def create_encoder(name, model, tokenizer, prompt_tokens, length=None, encoder_type="lstm"):
    embedding_dim = model.config.hidden_size
    cur_list = tokenizer.additional_special_tokens
    my_specials = [x for x in cur_list if not "<extra_id"  in x]
    if "@" in name:
        name, encoder_type = name.split("@") 

    prompt_encoder = None
    if encoder_type.startswith("mlp"):
        _enc_type = encoder_type.split("@")
        num_layers = 1
        if len(_enc_type) > 1:
            num_layers = int(_enc_type[1])
        hidden_size = -1
        if len(_enc_type) > 2:
            hidden_size = int(_enc_type[2])
        prompt_encoder = MLPPromptEncoder(name = name,
                model=model, tokenizer=tokenizer,
                prompt_tokens=prompt_tokens, 
                length = length,
                num_layers=num_layers, 
                hidden_size=hidden_size)
    elif encoder_type.startswith("emb"):
        prompt_encoder = EmbeddingPromptEncoder(name = name,
                model=model, tokenizer=tokenizer,
                length = length,
                prompt_tokens=prompt_tokens) 

    elif encoder_type.startswith("mat"):
        prompt_encoder = MatPromptEncoder(
                prefix_config=prefix_config, 
                name = name, 
                length = length,
                model=model, tokenizer=tokenizer,
                prompt_tokens=prompt_tokens) 
    else:
        _enc_type = encoder_type.split("@")
        num_layers = 1
        hidden_size = -1
        if len(_enc_type) > 1:
            num_layers = int(_enc_type[1])
        if len(_enc_type) > 2:
            hidden_size = int(_enc_type[2])
        prompt_encoder = LSTMEmbeddingPromptEncoder(
                name = name, 
                model=model, tokenizer=tokenizer,
                prompt_tokens=prompt_tokens, 
                length = length,
                num_layers=num_layers, 
                hidden_size=hidden_size)
    return prompt_encoder, encoder_type


