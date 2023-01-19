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

def _isin(tensor:torch.Tensor,values:torch.Tensor):
    return (tensor[..., None] == values).any(-1)

class PromptEncoder(torch.nn.Module):
    enc_type = "encoder"
    def __init__(self,name, embedding_dim, id_offset, init_embs, prompt_ids, lr=0.01, n_tasks = 2, router=None, **kwargs) -> None:
        super().__init__()
        self.learning_rate = lr
        self.length = length = len(prompt_ids)
        self.name = name
        self.task_id = -1 
        self.device = "cpu"
        self.counter = 0
        self.prompt_ids = prompt_ids
        self.input_ids = torch.nn.parameter.Parameter(torch.tensor(prompt_ids),
             requires_grad=False)
        self.net_inps = torch.nn.parameter.Parameter(torch.arange(length),
            requires_grad=False)
        self.embedding_dim = embedding_dim
        self.id_offset = id_offset
        self.embedding = torch.nn.Embedding(length,embedding_dim)
        self.init_embs = init_embs
        self.init_flag = True
        self.temperature = 1.
        #self.embedding.weight.requires_grad = False
        self.init_embedding(init_embs)
        para = [p for p in self.parameters() if p.requires_grad ]
        self.n_tasks = n_tasks
        self.is_learned = False
        if router == "random":
            self.is_learned = True
            self.router = nn.Parameter(data=torch.empty((
                self.n_tasks,
                length,
            )).uniform_(-1e-3, 1e-3))
        else:
            self.router = router

    def get_id(self):
        return "prompt_" + self.enc_type + "_" + self.name + "_" + str(self.length) + ".pt"

    def save(self, save_dir):
        if not save_dir:
            return
        fname = os.path.join(save_dir, self.get_id())
        torch.save(self.state_dict(), fname)

    def load(self, load_dir):
        if not load_dir:
            return
        fname = os.path.join(load_dir, self.get_id())
        if Path(fname).is_file():
            state = torch.load(fname)
            self.load_state_dict(state)
            mylogs.tinfo("Prompt for %s was loaded ", self.name)
            assert False, ""


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

    def freeze_router():
        self.reouter.requires_grad = False

    def unfreeze_router():
        self.router.requires_grad = True

    def forward(self,prompt_token_ids, tids=None, training=True):
        if tids is not None:
            task_id = tids[0]
        if self.id_offset > 0:
            index_list = prompt_token_ids - self.id_offset
        else:
            index_list = (prompt_token_ids.view(-1,1) == self.input_ids).int().argmax(dim=1)
        ret_embeds = self.forward_step(index_list, tids, training)
        return ret_embeds

    def forward_step(self, index_list, tids=None, training=True):
        raise NotImplementedError()

    def learn_router(self, tids=None, training=True):
        if self.router is None:
            return None
        if tids is not None:
            task_id = tids[0]
        return self.router
        router = self.router[task_id] # torch.index_select(self.router, 0, tids)
        if training and (not self.router.requires_grad or not self.is_learned):
            return router
        if self.init_flag:
            mylogs.tinfo("Initial Router: %s", self.router)
            self.init_flag = False
        if training:
            router = RelaxedBernoulli(temperature=self.temperature, logits=router).rsample()            # layer * n_prompts
        else:
            if logs.args("trunc_router") == "sigmoid":
                mylogs.tinfo("Trunc:===================TRUNC=====Sigmoid===========")
                router = torch.sigmoid(router)  # layer * n_prompts
            elif logs.main_args["trunc_router"] == "sign":
                with torch.no_grad():
                    mylogs.tinfo("Trunc:===================TRUNC======SIGN======")
                    mylogs.tinfo("Router Before relu: %s", router)
                    router[router <= 0] = 0
                    router[router > 0] = 1
                    mylogs.tinfo("Router After relu: %s", router)
        router = (router / (router.sum(dim=-1, keepdim=True) + 1e-12))  
        # layer * 1 * n_prompts
        return router

    def isin(self, ar1, ar2):
        return (ar1[..., None] == ar2).any(-1)
    def get_prompt_token_fn(self):
        if self.input_ids is not None:
            return lambda x: self.isin(x, self.input_ids)
        else:
            return lambda x: (x>=self.id_offset)&(x<self.id_offset+self.length)

    def dump_embeddings_into(self, weight, task_ids = None):
        mylogs.tinfo("%s ) Final Router (before forward): %s", self.name, self.router)
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
        router = self.learn_router(tids)
        self.embedding.wight = torch.mul(router.unsqueeze(1), self.embedding.weight).view(-1, self.embedding_dim)
        ret_embeds = self.embedding(index_list)
        return ret_embeds 

class MatPromptEncoder(PromptEncoder):
    def __init__(self, prefix_config, **kwargs):
        super().__init__(**kwargs)
        self.prefix_config = prefix_config
        if prefix_config is not None:
            self.temperature = self.prefix_config['temperature']

            self.router = nn.Parameter(data=torch.empty((
                prefix_config['n_tasks'],
                prefix_config['n_prompts']
            )).uniform_(-1e-3, 1e-3))

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
        router = self.learn_router(tids)
        z = torch.mm(self.z, self.A) if not hasattr(self, 'prompt') else self.prompt
        #ret_embeds = torch.matmul(router.unsqueeze(0), z).view(-1, self.embedding_dim)
        running_weight = torch.matmul(router, z).view(-1, self.embedding_dim)
        ret_embeds = F.embedding(index_list, running_weight)
        return ret_embeds 

class MergePromptEncoderBase(PromptEncoder):
    pass

class MergePromptEncoderOld(MergePromptEncoderBase):
    def __init__(self, encoders = [], trunc_router=False, wandb=False, **kwargs):
        super().__init__(**kwargs)
        self.task_id = 0
        self.temperature = 1 
        self.n_prompts = int(logs.main_args["prompt_token_num"]) #len(encoders) 
        self.n_tasks = 2
        self.flag = True
        self.flat = False
        self.trunc_router = trunc_router
        self.wandb = wandb
        self.set_encoders(encoders)
        self.router = nn.Parameter(data=torch.empty((
            self.n_tasks,
            self.n_prompts
        )).uniform_(-1e-3, 1e-3))

    def set_encoders(self, encoders):
        if encoders:
            self.encoders = torch.nn.ModuleList(encoders)
    def forward(self, prompt_token_ids, tids=None, training=True):
        device = self.device
        task_id = tids[0]
        assert task_id == 0 or logs.main_args["rel_filter"] == "", "Check task id " + str(task_id) + " rel:" + logs.main_args["rel_filter"]
        if self.wandb:
            wandb.log({'tid': task_id})
        if self.flag:
            mylogs.tinfo("Initial Router: %s", self.router)
            self.flag = False
        #mylogs.tinfo("Task ids: %s", tids)
        router = self.router[task_id]
        if training:
            router = RelaxedBernoulli(temperature=self.temperature, logits=router).rsample()  # layer * n_prompts
            router = (router / (router.sum(dim=-1, keepdim=True) + 1e-12))  
        else:
            #router = torch.sigmoid(router)  # layer * n_prompts
            if self.trunc_router:
                with torch.no_grad():
                    mylogs.tinfo("Router:========================================")
                    mylogs.tinfo("Router Before relu: %s", router)
                    router[router <= 0] = 0
                    router[router > 0] = 1
                    mylogs.tinfo("Router After relu: %s", router)
            #router = (router / (router.sum(dim=-1, keepdim=True) + 1e-12))  
        # layer * 1 * n_prompts
        #ret_embeds = torch.matmul(router.unsqueeze(0), z).view(-1, self.embedding_dim)
        if self.id_offset > 0:
            index_list = prompt_token_ids - self.id_offset
        else:
            index_list = (prompt_token_ids.view(-1,1) == self.input_ids).int().argmax(dim=1)
        z = torch.zeros(len(self.prompt_ids), self.embedding_dim).to(device)
        tl = []
        for encoder in self.encoders:
            pids = encoder.input_ids
            out = encoder(pids).to(device) 
            tl.append(out)
            z += out
        if self.flat:
            z = z / len(self.encoders)
        else:
            z = torch.vstack(tl) 
        z = z.view(self.n_prompts, -1) 
        if self.flat:
            running_weight = torch.mul(router.unsqueeze(1), z).view(-1, self.embedding_dim)
        else:
            running_weight = torch.matmul(router, z).view(-1, self.embedding_dim)
        ret_embeds = F.embedding(index_list, running_weight)
        return ret_embeds 

    def dump_embeddings_into(self, weight, task_ids = None):
        mylogs.tinfo("Final Router (before forward): %s", self.router)
        if task_ids == None:
            task_ids = [0]
        mylogs.tinfo("Gen ids %s", task_ids)
        with torch.no_grad():
            embs= self.forward(self.input_ids, tids=task_ids, training=False)
            detached_embeddings = embs.detach()
            weight[self.prompt_ids,:]=detached_embeddings

class MergePromptEncoder(MergePromptEncoderBase):
    def __init__(self, encoders = [], trunc_router=False, wandb=False, **kwargs):
        super().__init__(**kwargs)
        self.flag = True
        self.wandb = wandb
        self.set_encoders(encoders)

    def set_encoders(self, encoders):
        if encoders:
            self.encoders = torch.nn.ModuleList(encoders)
            self.n_prompts = len(encoders)

    def forward_step(self, index_list, tids=None, training=True):
        router = self.learn_router(tids, training)
        device = self.device
        tl = []
        z = torch.zeros((self.length, self.embedding_dim), device =self.device)
        for i, encoder in enumerate(self.encoders):
            pids = encoder.input_ids
            out = encoder(pids, tids).to(device) 
            z += router[i]*out
            tl.append(out)
        #z = torch.vstack(tl) 
        #z = z.view(self.length, -1) 
        #x = torch.mul(router.unsqueeze(1), z)
        #y = y.view(-1, self.embedding_dim)
        ret_embeds = F.embedding(index_list, z)
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
        router = self.learn_router(tids, training)
        embs = self.embedding(self.net_inps)
        z = self.mlp(embs)
        #z = z.view(self.length, -1) 
        #z = torch.mul(router.unsqueeze(1), z).view(-1, self.embedding_dim)
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
        if self.id_offset > 0:
            net_inputs = self.input_ids - self.id_offset
            #index_list = [((net_inputs == x).nonzero(as_tuple=True)[0]) for x in prompt_token_ids_2]
        # create embedding vectors for input ids
        embeds = self.embedding(net_inputs)
        x = self.lstm(embeds.unsqueeze(0))
        running_weight = self.mlp(x[0]).squeeze(0)
        if self.is_learned:
            router = self.learn_router(tids, training)
            z = running_weight.view(self.length, -1) 
            running_weight = torch.mul(router.unsqueeze(1), z).view(-1, self.embedding_dim)
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
                rstrip=False)
            for tok in new_tokens if not tok in cur_list
    ]
    if added_tokens:
        added_tokens = cur_list + added_tokens
        tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})

def create_encoder(name, model, tokenizer, prompt_tokens, 
        encoder_type="lstm", enc_router=None):
    embedding_dim = model.config.hidden_size
    enc_plen = len(prompt_tokens)

    task_tokens = prompt_tokens #+ common_tokens
    cur_list = tokenizer.additional_special_tokens
    my_specials = [x for x in cur_list if not "<extra_id"  in x]

    enc_plen =len(task_tokens) 
    mylogs.tinfo("** len tokenizer before extend: %s", len(tokenizer))
    extend_tokenizer(tokenizer, task_tokens)
    model.resize_token_embeddings(len(tokenizer))
    prompt_ids = tokenizer.convert_tokens_to_ids(task_tokens)
    cur_embeddings = model.get_input_embeddings()
    init_embs = {}
    mylogs.bp("encoder")
    if "@" in name:
        name, encoder_type = name.split("@") 

    for pid, p in enumerate(prompt_ids):
        if pid < cur_embeddings.num_embeddings:
            emb = cur_embeddings.weight[pid,:].detach().clone() 
            init_embs[pid] = emb

    for pid, p in enumerate(task_tokens):
        if "_" in p:
           q = p.strip("<").strip(">")
           w = q.split("_")[1]
           if not w.isdigit():
               wid = tokenizer.convert_tokens_to_ids([w])[0]
               emb = cur_embeddings.weight[wid,:].detach().clone() 
               init_embs[pid] = emb

    id_offset = min(prompt_ids) 
    prompt_encoder = None
    if encoder_type.startswith("mlp"):
        _enc_type = encoder_type.split("@")
        num_layers = 1
        if len(_enc_type) > 1:
            num_layers = int(_enc_type[1])
        hidden_size = -1
        if len(_enc_type) > 2:
            hidden_size = int(_enc_type[2])
        if enc_plen > 0:
            prompt_encoder = MLPPromptEncoder(name = name,
                    embedding_dim = embedding_dim,
                    id_offset = -1, prompt_ids=prompt_ids, init_embs=init_embs, num_layers=num_layers, hidden_size=hidden_size, router=enc_router)
    elif encoder_type.startswith("emb"):
        if enc_plen > 0:
            prompt_encoder = EmbeddingPromptEncoder(name = name,
                    embedding_dim = embedding_dim,
                    id_offset = -1, 
                    prompt_ids=prompt_ids, init_embs=init_embs, router=enc_router)
    elif encoder_type.startswith("mold"):
        if enc_plen > 0:
            prompt_encoder = MergePromptEncoderOld(name = name, 
                    embedding_dim = embedding_dim,
                    id_offset = -1, prompt_ids=prompt_ids, init_embs=init_embs, 
                    router=enc_router)
    elif encoder_type.startswith("merge"):
        if enc_plen > 0:
            prompt_encoder = MergePromptEncoder(name = name, 
                    embedding_dim = embedding_dim,
                    id_offset = -1, prompt_ids=prompt_ids, init_embs=init_embs, 
                    router=enc_router)
    elif encoder_type.startswith("mat"):
        if enc_plen > 0:
            prompt_encoder = MatPromptEncoder(prefix_config=prefix_config, 
                    name = name, 
                    embedding_dim = embedding_dim,
                    id_offset = -1, prompt_ids=prompt_ids, init_embs=init_embs)
    else:
        if enc_plen > 0:
            _enc_type = encoder_type.split("@")
            num_layers = 1
            hidden_size = -1
            if len(_enc_type) > 1:
                num_layers = int(_enc_type[1])
            if len(_enc_type) > 2:
                hidden_size = int(_enc_type[2])
            prompt_encoder = LSTMEmbeddingPromptEncoder(name = name, 
                    embedding_dim = embedding_dim,
                    id_offset = -1, prompt_ids=prompt_ids, init_embs=init_embs, 
                    num_layers=num_layers, 
                    hidden_size=hidden_size, router=enc_router)
    return prompt_encoder, encoder_type


