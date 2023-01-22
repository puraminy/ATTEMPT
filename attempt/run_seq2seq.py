# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
from utils import * 
import shutil
from pathlib import Path
import glob
from data import AutoPostProcessor
from third_party.models import T5Config, T5ForConditionalGeneration
from dataclasses import dataclass, field
from options import AdapterTrainingArguments, ModelArguments, DataTrainingArguments, TrainingArguments
from third_party.trainers import Seq2SeqTrainer
from data import TaskDataCollatorForSeq2Seq
from data import AutoTask
import re
from rouge import Rouge
from utils import get_adapter_config
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed,
    AdamW,
    get_linear_schedule_with_warmup
)
import transformers
from datasets import concatenate_datasets
from typing import Optional, List
import subprocess
import sys
import functools
import logging
import numpy as np
from pytz import common_timezones
import torch
import os
from torch import nn

from data.tasks import TASK_MAPPING
from metrics.metrics import TASK_TO_METRICS
from metrics.metrics import build_compute_metrics_fn

###### my imports
from myds import my_interleave_datasets
from conflicts import check_conflicts
import json
import glob
import mylogs 
import itertools, collections
from metrics.metrics import do_score
from encoders.encoders import *
from optim import *
import wandb

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

logger = logging.getLogger(__name__)

def mbp(bp="all",*arg):
    print("info:",*arg)
    mylogs.bp(bp)

def run_command(command):
    output = subprocess.getoutput(command)
    return output

import click
import debugpy
import os.path as op

@click.group()
def cli():
    pass
@cli.command(context_settings=dict(
            ignore_unknown_options=True,
            allow_extra_args=True,))
@click.option(
    "--experiment",
    "-exp",
    default="exp",
    type=str,
    help="Experiment name"
)
@click.option(
    "--exp_conf",
    "-cfg",
    default="",
    type=str,
    help="A file containing configs"
)
@click.option(
    "--break_point",
    "-bp",
    default="",
    type=str,
    help="Stop on breakpoints equal to the value"
)
@click.option(
    "--preview",
    "-pv",
    default="",
    type=str,
    help="The name of an experiment variable for which you want to check the difference of its values"
)
@click.option(
    "--debug",
    "-dpy",
    is_flag=True,
    help="Enable debugpy"
)
@click.option(
    "--trial",
    "-t",
    default="1",
    type=str,
    help="You can set it for repeating experiments with different identities"
)
@click.option(
    "--rem",
    "-rem",
    is_flag=True,
    help="Remove the existing experiment folder"
)
@click.option(
    "--download_model",
    "-dm",
    is_flag=True,
    help="Whether download pretrained model or load it from a directory"
)
@click.pass_context
def run(ctx, experiment, exp_conf, break_point, preview, 
        debug, trial, rem, download_model):
   if debug:
       port = "1234"
       debugpy.listen(('0.0.0.0', int(port)))
       print("Waiting for client at run...port:", port)
       debugpy.wait_for_client()  # blocks execution until client is attached
   exclude_list = []
   exp_args = {}
   if exp_conf:
        with open(exp_conf) as f:
            exp_args = json.load(f)
   save_path = os.path.join(mylogs.logPath, experiment)
   if Path(save_path).exists() and rem:
       #if input("Are you sure you want to delete the experiment folder?") == "y":
       #shutil.rmtree(save_path)
       save_path = save_path.rstrip("/")
       dirs = glob.glob(save_path + '/*/')
       for d in dirs:
            shutil.rmtree(d)

   if Path(save_path).is_file():
       os.remove(save_path)
   Path(save_path).mkdir(exist_ok=True, parents=True)
   args = {}
   args["save_path"] = save_path
   args["load_path"] = "" 
   if not download_model:
       args["load_path"] = mylogs.pretPath 
   args["experiment"] = "!" + experiment 
   args["trial"] = trial
   args["break_point"] = break_point 
   args["preview"] = preview 
   tags = exp_args["tag"] if "tag" in exp_args else [] 
   full_tags = exp_args["full_tag"] if "full_tag" in exp_args else [] 
   if break_point:
       mylogs.setbp(break_point)

   output_dir = ""
   all_vars = [x.strip("--") for x in ctx.args]
   var_names = [x.split("=")[0] for x in all_vars]
   values = [x.split("=")[1].split("#") for x in all_vars]
   var_dict = {k:n for k,n in zip(var_names, values)} 
   for key,val in var_dict.items():
       multi = [item for item in val if re.match("multi-(.*)", item)]
       members = [x.strip("@") for x in val if not x in multi]
       if multi:
           ext = []
           for m in multi:
               _, l = m.split("-")
               l = len(members) if l == "all" else int(l)
               val.remove(m)
               comb = itertools.combinations(members, l)
               ext.extend(["@".join(c) for c in comb])
           val.extend(ext)
           var_dict[key] = val

   var_names = list(var_dict.keys())
   values = list(var_dict.values())

   tag_exclude = [vv.strip("!") for vv in var_names if vv.startswith("!")]
   var_names = [vv.strip("!") for vv in var_names]
   for vv, cc in zip(var_names, values):
       if len(cc) == 1:
           exclude_list.append(vv)
       if len(cc) > 1:
           full_tags.append(vv)
           if not vv in tag_exclude: tags.append(vv)
           if preview and not preview == "all": 
               if not vv in preview and not vv in tag_exclude:
                   var_names.remove(vv)
                   values.remove(cc)

   if preview and not preview in tags and not preview == "all": 
      print("Eror:", preview, " must be all or one of ", tags, " which have multiple values")
      return

   args["tag"] = tags 
   args["full_tag"] = full_tags 
   tot_comb = [dict(zip(var_names, comb)) for comb in itertools.product(*values)]
   ii = 0
   orig_args = args.copy()
   logger.info("Total experiments:%s", len(tot_comb))
   for comb in tot_comb:
       _output_dir = [output_dir]
       for var_name,var_item in comb.items():
           var_item =var_item 
           args[var_name]=var_item
           if not var_name in exclude_list:
               _output_dir.append(var_name + "=" + str(var_item))
       ii += 1
       args["expid"] = ii if not "expid" in exp_args else exp_args["expid"]
       args = {**exp_args, **args}
       _output_dir.append(str(args["expid"]))
       args["output_dir"] = "!" + os.path.join(save_path, 
                                         args["method"] + "-" + args["trial"], 
                                         *_output_dir)
       # break point before running to check arguments (breakpoint must be check)
       mylogs.bp("check")
       wandb.init(
          # Set the project where this run will be logged
          project= experiment.replace("#","-") 
          name="@".join(tags)
          # Track hyperparameters and run metadata
          config=_ftag
       )
       ctx.invoke(train, **args)
       wandb.finish()

@cli.command()
def train(**kwargs):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    config_name = kwargs.setdefault("config","base")
    home = mylogs.home
    if config_name == "base":
        config_file =f"{home}/ATTEMPT/attempt/configs/baselines/base.json"
    elif config_name == "attempt":
        config_file= f"{home}/ATTEMPT/attempt/configs/attempt/single_task.json"

    exp_conf = json.dumps(kwargs, indent=2)
    mylogs.clog.info(exp_conf)
    preview = kwargs.setdefault("preview","")
    if preview == "all":
        print(exp_conf)
        return
    mylogs.set_args(kwargs.copy())
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
                               AdapterTrainingArguments))
    if config_file and config_file.endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=config_file)
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    #### My code: overwrite kwargs over arguments read from parser
    def overwrite_conf(kwargs):
        new_kwargs = {}
        for k,v in kwargs.items():
            logger.info("ARGS: %s=%s", k, v)
            v = strval(v)
            new_kwargs[k] = v
            if hasattr(model_args,k):
                setattr(model_args, k, v)
            elif hasattr(data_args,k):
                setattr(data_args, k, v)
            elif hasattr(training_args,k):
                setattr(training_args, k, v)
            elif hasattr(adapter_args,k):
                setattr(adapter_args, k, v)
        return new_kwargs

    # sssssssss
    training_args.report_to = ["tensorboard"]
    kwargs = overwrite_conf(kwargs)
    kwargs = dotdict(kwargs)
    exp_conf = json.dumps(kwargs, indent=2)
    print("============ CONF ===========")
    print(exp_conf)
    Path(training_args.output_dir).mkdir(exist_ok=True, parents=True)
    with open(op.join(training_args.output_dir,"exp.json"), "w") as f:
        print(exp_conf, file=f)
    mylogs.bp("conf")

    trainer_shuffle = kwargs.setdefault("trainer_shuffle", False)
    bp = kwargs.setdefault("break_point","")
    # set other options
    if type(data_args.task_name) != list:
        data_args.task_name = [data_args.task_name]
    data_args.eval_dataset_name=data_args.task_name
    data_args.test_dataset_name=data_args.task_name

    task_args = {}
    task_args["data_seed"] = data_args.data_seed
    task_args["train_samples"] = data_args.max_train_samples
    task_args["val_samples"] = data_args.max_val_samples
    task_args["test_samples"] = data_args.max_test_samples
    task_args["num_prompt_tokens"] = adapter_args.num_prompt_tokens
    task_args["num_common_tokens"] = adapter_args.num_common_tokens
    task_args["template"] = data_args.template
    task_args["data_path"] = data_args.data_path
    task_args["rels"] = kwargs.rels
    task_args = dotdict(task_args)

    # an option to explicitly specify the method of training 
    # (pt: prompt-tuning, ft:fine-tuning, px:prefix-tuning etc.)
    method = kwargs.setdefault("method", "")
    ds_confs = kwargs.setdefault("ds_config", ["en"])
    n_tasks = len(data_args.task_name)
    _confs =["en"] * n_tasks
    for i,c in enumerate(ds_confs):
        _confs[i] = ds_confs[i]
    if kwargs.setdefault("adjust_epochs", False):
        training_args.num_train_epochs *= n_tasks

    data_args.dataset_config_name = _confs
    data_args.eval_dataset_config_name = _confs

    test_ds_confs = kwargs.setdefault("test_ds_config", ["test"])
    test_ds_names = data_args.test_dataset_name
    mylogs.bp("conf")
    test_combs = itertools.product(test_ds_confs, test_ds_names)
    _confs = []
    _names = []
    for c, n in test_combs:
        _confs.append(c)
        _names.append(n)
    data_args.test_dataset_name = _names
    data_args.test_dataset_config_name = _confs

    #if type(data_args.task_name) == list:
    #    model_args.multi_task = True

    # tags are variables that are varied among experiments. 
    tag = kwargs.setdefault("tag",[]) # the selected tags
    full_tag = kwargs.setdefault("full_tag",[]) # the full list of tags
    # check conflicts of options
    check_cfls = kwargs.setdefault("check_conflicts",True)
    if check_cfls: #check conflicts
        resolved, msg = check_conflicts(model_args, data_args, 
                training_args, adapter_args, kwargs)
        print(msg)
        title = mylogs.get_tag(full_tag)
        title = json.dumps(title, indent=4)
        mylogs.dlog.info(title)
        mylogs.dlog.info("%s", msg)
        mylogs.dlog.info("-------------------------------------")
        if not resolved:
            return

    if preview:
       mylogs.plog.handlers.clear()
       mylogs.add_handler(mylogs.plog, preview + "_" + str(kwargs[preview]))
       mylogs.plog.info(exp_conf)
    ###### Collect experiment infos
    exp_info = {}
    for k,v in kwargs.items():
        if not k in exp_info:
            exp_info[k] = v

    _tag = mylogs.get_tag(tag)  
    exp_info["tag"] = list(_tag.values())
    exp_info["taginfo"] = list(_tag.keys())
    _ftag = mylogs.get_tag(full_tag)  
    exp_info["ftag"] = _ftag 
    ######
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("#### last_checkpoint ", last_checkpoint)
        if training_args.resume_from_checkpoint is None or (last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0):
            '''
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
            '''
            existing_results = glob.glob(op.join(training_args.output_dir, "*.tsv"))
            if existing_results and not preview:
                print("Skipping experiment:", training_args.output_dir)
                return 
            #last_checkpoint = None
            #out = training_args.output_dir
            #out += "_" + mylogs.now
            #Path(out).mkdir(parents = True, exist_ok=True)
            #training_args.output_dir = out
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load a model config
    model_name_or_path =  model_args.config_name if model_args.config_name else model_args.model_name_or_path
    load_path = kwargs.setdefault("load_path", "")
    if not model_name_or_path.startswith("/") and load_path:
        model_name_or_path = op.join(load_path, model_name_or_path)
    config = T5Config.from_pretrained(
        model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.train_task_adapters = adapter_args.train_task_adapters
    config.prefix_tuning = adapter_args.prefix_tuning
    config.prompt_tuning = adapter_args.prompt_tuning
    config.attn_tuning = model_args.attn_tuning
    config.attn_method = model_args.attn_method
    config.ignore_target = model_args.ignore_target
    config.shared_attn = model_args.shared_attn
    if model_args.prompt_embedding_path:
        config.prefix_num = len(model_args.prompt_embedding_path) 
    else:
        config.prefix_num = model_args.prefix_num
    config.num_target = len(data_args.task_name)
    config.temperature = model_args.temperature
    config.learned_temperature = model_args.learned_temperature
    config.fix_attention = model_args.fix_attention
    adapter_config = get_adapter_config(
        adapter_args, data_args, training_args, config)

    # Set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Initialize the model
    model = T5ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        #cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        adapter_config=adapter_config
    )

    mapl=torch.device('cpu')
    if model_args.load_prefix_embeddings is True:
        if model_args.prompt_embedding_path is None:
            for name, param in model.named_parameters():
                if "prefix_shared" in name or "prefix" in name:
                    shared_params = [param]
        else:
            shared_params = []
            for rel_path in model_args.prompt_embedding_path:
                path = op.join(mylogs.pretPath, "prefixes", rel_path) 
                shared_param = torch.load(path, map_location=mapl)
                shared_params.append(shared_param)
            if model_args.target_prompt_embedding_path is not None:
                target_prompt_embedding = torch.load(
                    model_args.target_prompt_embedding_path, map_location=mapl)

        if model_args.attn_tuning is True:
            if training_args.do_train is True and model_args.multi_task is False and model_args.shared_attn is False:
                # Initialize the prompt embeddings using the first prompts
                # Load all of the target prompts
                model.store_prefix_weights(shared_params)
                model.update_prefix_weights_single(shared_params[0])
            elif training_args.do_train is True and model_args.multi_task is False and model_args.shared_attn is True:
                # initialize the embeddings
                # initialize multiple shared embeddings
                model.store_prefix_weights(shared_params)
                model.update_prefix_weights_multi(
                    shared_params[0], num_target=config.num_target)
            else:
                # Load prompt embeddings except for the last one
                # Load last prompt embeddings to initialize the target prompt embeddings.
                model.store_prefix_weights(shared_params)
                model.update_prefix_weights_single(shared_params[-1])

        else:
            if model_args.target_prompt_embedding_path is None:
                model.update_prefix_weights(shared_params)
            else:
                model.update_prefix_weights(
                    shared_params, target_prompt_embedding)

    if model_args.load_attention is True and model_args.attn_path is not None:
        model.update_attention_weights(torch.load(model_args.attn_path, map_location=mapl))

    if model_args.load_attention is True and model_args.attn_path_sub is not None:
        model.update_attention_weights_sub(model_args.attn_path_sub)

    if model_args.load_layer_norm is True and model_args.layer_norm_dir is not None:
        model.update_layer_norm_weights(model_args.layer_norm_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ######################## My code pppppp
    prompts_dir = model_args.prompt_encoders_dir
    if prompts_dir and not prompts_dir.startswith("/"):
        prompts_dir = op.join(mylogs.pretPath, prompts_dir) 
    if adapter_args.prompt_tuning:
        added = add_specials(tokenizer)
        logger.info("%s tokens was addded", added)
        if bp and bp in "tokens|encoder": breakpoint()
        model.resize_token_embeddings(len(tokenizer))
        # mmmmmmmmmmmmm
        prompts = {}
        mylogs.bp("prompts")
        if model_args.attn_tuning:
            assert data_args.source_tasks,"Source tasks can't be empty for attention tuning"
            if list(set(data_args.source_tasks) & set(data_args.task_name)) != []: 
                 raise ValueError("Source tasks shoudn't intersect with target tasks")
        tasks = data_args.task_name
        n_tasks = len(tasks)
        for task in tasks:
             bp != "prompts" or breakpoint()
             p = AutoTask.get(task, None, task_args=task_args).get_prompts()
             prompts = {**prompts, **p}
        ii = 0
        load_prompts = kwargs.setdefault("load_prompts", False) 
        prompt_encoders = []
        for task, prompt_tokens in prompts.items():
            enc_router = torch.ones((n_tasks, len(prompt_tokens)), 
                    requires_grad=False, device=device)
            encoder, enc_type = create_encoder(task, model, tokenizer, 
                    prompt_tokens, adapter_args.prompt_encoder_type, 
                    enc_router = enc_router)
            if kwargs.setdefault("init_from_words", False):
                encoder.init_embs_from_words(model.get_input_embeddings())
            if load_prompts and (not model_args.attn_tuning 
                                 or task in data_args.source_tasks):
                encoder.load(prompts_dir)
            prompt_encoders.append(encoder)
            if not "com" in task: # if it's not a shared prompt among tasks
                encoder.task_id = ii
                ii += 1
        model.encoder.set_encoders(prompt_encoders, 
            data_args.source_tasks, adapter_args.num_prompt_tokens, load_prompts)
        model.resize_token_embeddings(len(tokenizer))
        if bp == "tokens": breakpoint()

    rgrad = len([p for p in model.parameters() if p.requires_grad])
    nrgrad = len([p for p in model.parameters() if not p.requires_grad])
    mylogs.plog.info("Before freeze: requires grad: %s   Not requires grad: %s", rgrad, nrgrad)
    model = modify_model_after_init(
        model, training_args, adapter_args, adapter_config)

    rgrad = len([p for p in model.parameters() if p.requires_grad])
    nrgrad = len([p for p in model.parameters() if not p.requires_grad])
    mylogs.plog.info("After freeze: requires grad: %s   Not requires grad: %s", rgrad, nrgrad)
    mylogs.bp("freeze")

    data_args.dataset_name = data_args.task_name
    data_args.eval_dataset_name = data_args.eval_dataset_name
    data_args.test_dataset_name = data_args.test_dataset_name
    data_args.dataset_config_name = data_args.dataset_config_name
    data_args.eval_dataset_config_name = data_args.eval_dataset_config_name
    data_args.test_dataset_config_name = data_args.test_dataset_config_name
    assert len(data_args.dataset_name) == len(data_args.dataset_config_name)
    if data_args.eval_dataset_name is not None:
        assert len(data_args.eval_dataset_name) == len(
            data_args.eval_dataset_config_name)
    if data_args.test_dataset_name is not None:
        assert len(data_args.test_dataset_name) == len(
            data_args.test_dataset_config_name)

    # Temporarily set max_target_length for training.
    #max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    ########### rrrrrr
    def preprocess_function(examples, max_target_length, task_id=None):
        model_inputs = tokenizer(examples['source'], max_length=data_args.max_source_length,
                                 padding=padding, truncation=True)
        if preview:
            mylogs.plog.info("sourece: %s", examples["source"][:1])
            mylogs.plog.info("target: %s", examples["target"][:1])
        if bp and bp in "data|examples":
            logger.info("sourece: %s", examples["source"][:5])
            logger.info("target: %s", examples["target"][:5])
            logger.info("extra: %s", examples["extra_fields"][:5])
            breakpoint()
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target'], max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        if "task_ids" in examples["extra_fields"]:
            model_inputs["task_ids"] = examples["extra_fields"]["task_ids"]
        model_inputs["extra_fields"] = examples['extra_fields']  
        if task_id is not None:
            model_inputs["task_ids"] = [
                task_id for _ in examples['extra_fields']]
        return model_inputs

    column_names = ['source', 'target', 'extra_fields']
    performance_metrics = {}
    if training_args.do_train:
        # Load datasets from files if your target datasets are not in huggingface datasets.
        if data_args.train_files is not None:
            train_datasets = [AutoTask.get(dataset_name,
                                           dataset_config_name,
                                           task_args=task_args).get(
                split="train",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_train_samples, lang=data_args.lang_name, file_name=train_file)
                for dataset_name, dataset_config_name, train_file
                in zip(data_args.dataset_name, data_args.dataset_config_name, data_args.train_files)]
        else:
            train_datasets = [AutoTask.get(dataset_name,
                                           dataset_config_name,
                                           task_args=task_args).get(
                split="train",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_train_samples, lang=data_args.lang_name, file_name=data_args.train_file)
                for dataset_name, dataset_config_name
                in zip(data_args.dataset_name, data_args.dataset_config_name)]

        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name, task_args=task_args).get_max_target_length(
            tokenizer=tokenizer, default_max_length=data_args.max_target_length, )
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)]

        for i, train_dataset in enumerate(train_datasets):
            if model_args.shared_attn is True:
                train_datasets[i] = train_datasets[i].map(
                    functools.partial(
                        preprocess_function, max_target_length=max_target_lengths[i], task_id=i),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # if train_dataset != "superglue-record" else column_names+["answers"],
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
            else:
                train_datasets[i] = train_datasets[i].map(
                    functools.partial(preprocess_function,
                                      max_target_length=max_target_lengths[i]
                                      #mycode adding task ids
                                      ,task_id=i
                                      ),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # if train_dataset != "superglue-record" else column_names+["answers"],
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
        bp != "concat" or breakpoint()
        if trainer_shuffle:
            train_dataset = concatenate_datasets(train_datasets)
        else:
            train_dataset = my_interleave_datasets(train_datasets, 
                batch_size=training_args.per_device_train_batch_size)

    if training_args.do_eval:
        if data_args.validation_files is not None:
            eval_datasets = {eval_dataset: AutoTask.get(eval_dataset, eval_dataset_config,
                                                        task_args=task_args).get(
                split="validation",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_val_samples, lang=data_args.lang_name, file_name=validation_file)
                for eval_dataset, eval_dataset_config, validation_file in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name, data_args.validation_files)}
        else:
            eval_datasets = {eval_dataset: AutoTask.get(eval_dataset, eval_dataset_config,
                                                        task_args=task_args).get(
                split="validation",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_val_samples, lang=data_args.lang_name, file_name=data_args.validation_file)
                for eval_dataset, eval_dataset_config in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)}

        max_target_lengths = [AutoTask.get(dataset_name, 
            dataset_config_name,
            task_args=task_args).get_max_target_length(
            tokenizer=tokenizer, default_max_length=data_args.max_target_length)
            for dataset_name, dataset_config_name in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)]

        for k, name in enumerate(eval_datasets):
            if model_args.shared_attn is True:
                eval_datasets[name] = eval_datasets[name].map(
                    functools.partial(
                        preprocess_function, max_target_length=max_target_lengths[k], task_id=k),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # if name != "superglue-record" else column_names+["answers"],
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
            else:
                eval_datasets[name] = eval_datasets[name].map(
                    functools.partial(preprocess_function,
                                      max_target_length=max_target_lengths[k]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # if name != "superglue-record" else column_names+["answers"],
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )

    if training_args.do_test:
        if data_args.test_files is not None:
            test_datasets = {test_dataset + "_" + test_dataset_config: AutoTask.get(test_dataset, test_dataset_config,
                                                        task_args=task_args).get(
                split="test",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_test_samples, lang=data_args.lang_name, file_name=test_file)
                for test_dataset, test_dataset_config, test_file in zip(data_args.test_dataset_name, data_args.test_dataset_config_name, data_args.test_files)}
        else:
            mylogs.bp("test")
            test_datasets = {test_dataset + "_" + test_dataset_config: AutoTask.get(test_dataset, test_dataset_config,
                                                        task_args=task_args).get(
                split="test",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_test_samples, lang=data_args.lang_name, file_name=data_args.test_file)
                for test_dataset, test_dataset_config in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)}

        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name,
            task_args=task_args).get_max_target_length(
            tokenizer=tokenizer, default_max_length=data_args.max_target_length)
            for dataset_name, dataset_config_name in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)]
        for k, name in enumerate(test_datasets):
            if model_args.shared_attn is True:
                test_datasets[name] = test_datasets[name].map(
                    functools.partial(
                        preprocess_function, max_target_length=max_target_lengths[k], task_id=k),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
            else:
                test_datasets[name] = test_datasets[name].map(
                    functools.partial(preprocess_function,
                                      max_target_length=max_target_lengths[k]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )

    if preview == "template":
        return
    # Data collator
    label_pad_token_id = - \
        100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = TaskDataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    eval_metrics = [AutoTask.get(dataset_name, 
                    dataset_config_name, task_args=task_args).metric
                    for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)][0]

    print(data_args.eval_dataset_name)
    compute_metrics_fn = build_compute_metrics_fn(
        data_args.eval_dataset_name, tokenizer, data_args.ignore_pad_token_for_loss) if training_args.predict_with_generate else None
    print(compute_metrics_fn)

    data_info = {}
    has_extra = kwargs.setdefault("has_extra", True)
    if has_extra:
        data_info["eval"] = eval_datasets[data_args.eval_dataset_name[0]]['extra_fields'] if training_args.do_eval else None
        data_info["test"] = test_datasets[data_args.test_dataset_name[0] + "_" + data_args.test_dataset_config_name[0]]['extra_fields'] if training_args.do_test else None
        data_info["train"] = train_dataset['extra_fields'] if training_args.do_train else None

    def compute_metrics(eval_preds):
        preds, labels, data_info = eval_preds
        post_processor = AutoPostProcessor.get(data_args.dataset_name[0], tokenizer,
                                               data_args.ignore_pad_token_for_loss)
        decoded_preds, decoded_labels = post_processor.process(
            preds, labels, data_info)
        result = {}
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result

    # If you want to use a different learning rate for attention layer, initialize an optimizer using the learning rate here.
    if bp == "opt": breakpoint()
    grouped_params = []
    all_parameters = set([p for p in model.parameters() if p.requires_grad])
    attn_params = []
    if model_args.attn_learning_rate is not None:
        for name, param in model.named_parameters():
            if name == "encoder.attn_W_up" or name == "encoder.attn_W_down" or name == "encoder.layer_norm":
                attn_params += list(param)
        attn_params = set(attn_params)
        grouped_params.append({'params': list(attn_params), 
            'lr': model_args.attn_learning_rate})
        

    ########### My Code
    prompt_params = []
    if adapter_args.prompt_tuning and model_args.prompt_learning_rate is not None:
        for encoder in model.prompt_encoders:
           para_list =[p for p in encoder.parameters() if p.requires_grad]
           prompt_params.extend(para_list)

        prompt_params = set(prompt_params)
        grouped_params.append({'params': list(prompt_params), 
            'lr': model_args.prompt_learning_rate})

    other_params = all_parameters - set(attn_params) - set(prompt_params)
    other_params = list(other_params)
    grouped_params.append({'params': other_params})
    #### ooooo 
    steps = 0
    if training_args.do_train:
        steps = len(train_dataset) * training_args.num_train_epochs // (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size)

    if kwargs.opt_type == "sep":
        optim, scheduler = get_optimizer(model, steps,
                model_args.prompt_learning_rate, 0.01, 0.01)
    else:
        optim = AdamW(grouped_params, lr=training_args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=training_args.warmup_steps, num_training_steps=steps)
    name = data_args.dataset_name[0] 
    task_metric = TASK_TO_METRICS[name] if name in TASK_TO_METRICS else ["rouge"]
    if training_args.do_eval: 
        eval_ds = list(eval_datasets.values())[0] 
    else: 
        eval_ds = None
    if kwargs.use_optimizer:
        # Initialize our Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset= eval_ds,
            data_info=data_info,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            multi_task_compute_metrics=compute_metrics_fn,
            evaluation_metrics=task_metric,
            save_checkpoint = kwargs.setdefault("save_checkpoint", False),
            shared=model_args.shared_attn,
            shuffle = trainer_shuffle,
            optimizers=(optim, scheduler)
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_ds,
            data_info=data_info,
            tokenizer=tokenizer,
            data_collator=data_collator,
            shuffle = trainer_shuffle,
            save_checkpoint = kwargs.setdefault("save_checkpoint", False),
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            evaluation_metrics=task_metric,
            multi_task_compute_metrics=compute_metrics_fn,
            shared=model_args.shared_attn)

    # Exit program if user wants to check some settings 
    if preview:
        return
    # Saves training config.
    if trainer.is_world_process_zero():
        os.makedirs(training_args.output_dir, exist_ok=True)
        save_training_config(config_file, training_args.output_dir)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        if training_args.compute_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"total_time in minutes ": total_time})

        # By setting the `save_prefix_only` True, you only save the attentions as well as the prompt components only.
        prefix_dir = model_args.prefix_dir
        if prefix_dir and not prefix_dir.startswith("/"):
            prefix_dir = op.join(mylogs.pretPath, prefix_dir) 
        if adapter_args.prompt_tuning or adapter_args.prefix_tuning:
            if prefix_dir:
                Path(prefix_dir).mkdir(parents = True, exist_ok=True)
            save_prompts(trainer.model, output_dir=training_args.output_dir, 
                         prefix_dir = prefix_dir,
                         attn_tuning=model_args.attn_tuning,
                         shared_attn=model_args.shared_attn, num_target=config.num_target, task_name=data_args.task_name)
        if adapter_args.prompt_tuning:
            Path(prompts_dir).mkdir(parents = True, exist_ok=True)
            model.store_encoders(output_dir = prompts_dir)

        if kwargs.setdefault("save_model", False):
            # save all model parameters and tokenizers 
            # regardless of whether they are updated or not.
            trainer.save_model()

        train_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        train_metrics["train_samples"] = min(
            max_train_samples, len(train_dataset))
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)

        if not model_args.save_prefix_only:
            trainer.save_state()

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        print(
            "Memory utilization",
            peak_memory,
            "GB"
        )
        performance_metrics.update({"peak_memory": peak_memory})
    if training_args.compute_memory or training_args.compute_time:
        trainer.save_metrics("performance", performance_metrics)

    # Validation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if model_args.attn_tuning is True:
            attention_paths = [os.path.join(training_args.output_dir, "attn_W_down.pt"), os.path.join(
                training_args.output_dir, "attn_W_up.pt")]
            trainer.model.update_attention_weights_sub(attention_paths)
            if model_args.load_layer_norm is True and "layer_norm_bias.pt" in training_args.output_dir:
                trainer.model.update_layer_norm_weights(
                    training_args.output_dir)

        if  model_args.shared_attn is False:
            for task, eval_dataset in eval_datasets.items():
                metrics = trainer.evaluate(eval_dataset=eval_dataset,
                                           max_length=data_args.val_max_target_length, num_beams=data_args.num_beams,
                                           )
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

    # Test
    mylogs.bp("test")
    if training_args.do_test:
        logger.info("*** Test ***")
        # multi-task evaluations
        results = {}
        if model_args.shared_attn is False:
            for idx, (task, test_dataset) in enumerate(test_datasets.items()):
                mylogs.bp("test")
                predictions, labels, metrics = trainer.predict(test_dataset=test_dataset,
                                           max_length=data_args.test_max_target_length, 
                                           num_beams=data_args.num_beams,
                                           metric_key_prefix="test"
                                           )
                
                trainer.log_metrics("test", metrics)
                trainer.save_metrics("test", metrics)
                ds_conf = data_args.test_dataset_config_name[idx]
                ds_name = data_args.test_dataset_name[idx]

                # sssssssssss
                #predictions = np.argmax(predictions, axis=1)
                #predictions = tokenizer.batch_decode(predictions)
                df = test_dataset.to_pandas()
                if bp == "test": breakpoint()
                df["pred_text1"] = ""
                #df["rouge_score"] = 0.0
                #df["bert_score"] = 0.0
                df["template"] = data_args.template
                df["resp"] = ""
                df["query"] = ""
                df["langs"] = "en2en"
                df["prefix"] = ds_name
                df["src_path"] = op.join(mylogs.home, data_args.data_path, 
                                        ds_conf,"test.tsv")
                for key, info in exp_info.items():
                    if type(info) == list:
                        info = "@".join(info)
                    if type(info) == dict:
                        info = json.dumps(info)
                        info = info.replace("\n", "@")
                    df[key] = info
                rouge_scorer = Rouge()
                for i, row in df.iterrows():
                    extra = row["extra_fields"]
                    if "event" in extra:
                        inp = extra["event"]
                    else:
                        inp = tokenizer.decode(row["input_ids"], 
                            skip_special_tokens=kwargs.setdefault("skip_spcials", True)) 
                        inp = re.sub(r'<.*?>','', inp)
                        inp = inp.strip()
                    df.at[i, "input_text"] = inp #extra["event"] 
                    try:
                        label = tokenizer.decode(row["labels"], 
                        skip_special_tokens=kwargs.setdefault("skip_spcials", True)) 
                    except:
                        label = extra["tail"] if "tail" in extra else "na"
                    #label = re.sub(r'<.*?>','', label)
                    label = label.strip()
                    df.at[i, "target_text"] = label 
                    sel = False
                    if "sel" in extra:
                        sel = extra["sel"] 
                    df.at[i, "sel"] = sel 
                    df.at[i, "resp"] = extra["resp"]  
                    pred = tokenizer.decode(predictions[i], 
                            skip_special_tokens=kwargs.setdefault("skip_spcials", True)) 
                    pred = re.sub(r'<.*?>','',pred)
                    pred = pred.strip()
                    df.at[i, "pred_text1"] = pred
                df.drop(columns=["input_ids","labels","attention_mask"])
                mylogs.bp("test")
                save_to = os.path.join(training_args.output_dir, 
                        ds_conf + "_results_" + ds_name + ".tsv")
                do_score(df, "rouge@bert", save_to)

            if kwargs.setdefault("eval_test", False):
                for task, test_dataset in test_datasets.items():
                    metrics = trainer.evaluate(eval_dataset=test_dataset,
                                               max_length=data_args.test_max_target_length, 
                                               num_beams=data_args.num_beams,
                                               metric_key_prefix="test"
                                               )
                    trainer.log_metrics("test", metrics)
                    trainer.save_metrics("test", metrics)

    if model_args.save_prefix_only:
        checkpoints = glob.glob(os.path.join(
            training_args.output_dir, "checkpoint-*"))
        for checkpoint_dir in checkpoints:
            # save models
            if not os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")):
                continue
            checkpoint_model = torch.load(os.path.join(
                os.path.join(checkpoint_dir, "pytorch_model.bin")))
            model.load_state_dict(checkpoint_model)
            new_dir = "{}_prompt_only".format(checkpoint_dir)
            os.mkdir(new_dir)
            save_prompts(model, output_dir=new_dir, 
                         prefix_dir = prefix_dir,
                         attn_tuning=model_args.attn_tuning,
                         shared_attn=model_args.shared_attn, num_target=config.num_target, task_name=data_args.task_name)
            if adapter_args.prompt_tuning:
                Path(op.join(new_dir, "prompts")).mkdir(parents = True, exist_ok=True)
                model.store_encoders(output_dir = prompts_dir)

            # after saving prompts, we will remove unnecessary checkpoint dir.
            try:
                shutil.rmtree(checkpoint_dir)
            except OSError as e:
                print("Error: %s : %s" % (checkpoint_dir, e.strerror))

    # Evaluate all checkpoints on all tasks if training_args.eval_all_at_last==True
    results = {}
    if training_args.eval_all_at_last:
        mylogs.bp("eval")
        for checkpoint_dir in glob.glob(os.path.join(training_args.output_dir, "checkpoint-*_prompt_only")):
            print(checkpoint_dir)
            mylogs.bp("eval")
            attention_paths = [os.path.join(checkpoint_dir, "attn_W_down.pt"), os.path.join(
                checkpoint_dir, "attn_W_up.pt")]
            trainer.model.update_attention_weights_sub(attention_paths)

            if model_args.load_layer_norm is True and "layer_norm_bias.pt" in checkpoint_dir:
                trainer.model.update_layer_norm_weights(checkpoint_dir)
            dev_metrics_all = {}
            dev_avg = []
            logger.info("*** Evaluate ***")
            for idx, (task, eval_dataset) in enumerate(eval_datasets.items()):
                if idx > 0:
                    print(task)
                    print(eval_metrics)
                shared_param = torch.load(os.path.join(
                    checkpoint_dir, "prefix_embeddings_{}.pt".format(data_args.task_name[idx])))
                trainer.model.update_prefix_weights_multi(
                    shared_param, num_target=1)
                metrics = trainer.evaluate(eval_dataset=eval_dataset,
                                           max_length=data_args.val_max_target_length, 
                                           num_beams=data_args.num_beams,
                                           )
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)
                dev_metrics_all[task] = metrics
                main_metric = list(metrics.values())[0]
                dev_avg.append(main_metric)

            results.setdefault(checkpoint_dir, {})
            results[checkpoint_dir]["dev_avg"] = np.mean(dev_avg)
            results[checkpoint_dir]["dev_each"] = dev_metrics_all

        # Test
        logger.info("*** Test ***")
        for checkpoint_dir in glob.glob(os.path.join(training_args.output_dir, "checkpoint-*_prompt_only")):
            # load models here
            attention_paths = [os.path.join(checkpoint_dir, "attn_W_down.pt"), os.path.join(
                checkpoint_dir, "attn_W_up.pt")]
            trainer.model.update_attention_weights_sub(attention_paths)
            if model_args.load_layer_norm is True and "layer_norm_bias.pt" in checkpoint_dir:
                trainer.model.update_layer_norm_weights(checkpoint_dir)

            test_metrics_all = {}
            test_avg = []
            for idx, (task, test_dataset) in enumerate(test_datasets.items()):
                shared_param = torch.load(os.path.join(
                    checkpoint_dir, "prefix_embeddings_{}.pt".format(data_args.task_name[idx])))
                trainer.model.update_prefix_weights_multi(
                    shared_param, num_target=1)
                metrics = trainer.evaluate(eval_dataset=test_dataset,
                                           max_length=data_args.test_max_target_length, 
                                           num_beams=data_args.num_beams,
                                           metric_key_prefix="test"
                                           )
                trainer.log_metrics("test", metrics)
                trainer.save_metrics("test", metrics)
                test_metrics_all[task] = metrics
                main_metric = list(metrics.values())[0]
                test_avg.append(main_metric)
            results.setdefault(checkpoint_dir, {})
            results[checkpoint_dir]["test_avg"] = np.mean(test_avg)
            results[checkpoint_dir]["test_each"] = test_metrics_all
    print(results)
    return results

if __name__ == "__main__":
   cli()
