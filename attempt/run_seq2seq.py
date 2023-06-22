# version 1400
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
from attempt.utils.utils import combine_x,combine_y
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from transformers import (
    AutoTokenizer,
    MT5TokenizerFast,
    T5TokenizerFast,
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
from callbacks import WBCallback, AnnealCallback, PTLearningRateCallback
import json
import pandas as pd
import glob
import mylogs 
import itertools, collections
from attempt.myutil import tag_to_image
from metrics.metrics import do_score
from encoders.encoders import *
from optim import *
from PIL import Image
import wandb
from deepdiff import DeepDiff

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
    type=str,
    help="Show only experiment configuraton or some data"
)
@click.option(
    "--exp_vars",
    "-ev",
    type=str,
    default="",
    help="The name of experiment multi-valued variables for which you want to check the difference of their values, if not given it runs all combinations"
)
@click.option(
    "--main_vars",
    "-mv",
    type=str,
    default="",
    help="The name of one multi-valued variable for which you want to check the difference of their values, if not given it runs all combinations"
)
@click.option(
    "--log_var",
    "-lv",
    type=str,
    default="",
    help="The name of an experiment multi-valued variables for which you want to log some data in a logfile names varied with the different values of the varibale"
)
@click.option(
    "--debug",
    "-d",
    default="",
    type=str,
    help="Enable debugpy, you can specify a breakpoint too"
)
@click.option(
    "--trial",
    "-t",
    default="1",
    type=str,
    help="You can set it for repeating experiments with different identities"
)
@click.option(
    "--version",
    "-v",
    default="1",
    type=str,
    help="You can set it for continueing experiments with different versions (after some changes)"
)
@click.option(
    "--rem",
    "-rem",
    is_flag=True,
    help="Remove the existing experiment folder"
)
@click.option(
    "--repeat",
    "-rep",
    is_flag=True,
    help="Repeat an experiment even if the folder already exists",
)
@click.option(
    "--download_model",
    "-mod",
    is_flag=True,
    help="Whether download pretrained model or load it from a directory"
)
@click.option(
    "--max_exp",
    "-max",
    default=0,
    type=int,
    help="Max number of experiments to do (0 means all)"
)
@click.option(
    "--new_exp_folder",
    "-new",
    is_flag=True,
    help="Whether create a new directory for experiment when loadign an existing config file"
)
@click.pass_context
def run(ctx, experiment, exp_conf, break_point, preview, exp_vars, log_var, main_vars, 
        debug, version, trial, rem, repeat, download_model, max_exp, new_exp_folder):
   if debug:
       port = "1234"
       if not break_point: break_point = debug
       debugpy.listen(('0.0.0.0', int(port)))
       print("Waiting for client at run...port:", port)
       debugpy.wait_for_client()  # blocks execution until client is attached
   exclude_list = []
   exp_args = {}
   save_path = ""
   if exp_conf:
        with open(exp_conf) as f:
            exp_args = json.load(f)
   experiment = experiment.replace("#","-").replace("@","-")
   if exp_conf: 
       save_path = exp_args["save_path"]
   if not save_path or experiment == "self":
       save_path = os.getcwd()
   if (not exp_conf and experiment != "self") or new_exp_folder:
       if new_exp_folder and save_path:
          save_path = os.path.join(str(Path(save_path).parent), experiment)
       else:
          save_path = os.path.join(mylogs.logPath, experiment)
       if Path(save_path).exists() and rem:
           #if input("Are you sure you want to delete the experiment folder?") == "y":
           #shutil.rmtree(save_path)
           save_path = save_path.rstrip("/")
           dirs = glob.glob(save_path + '/*/')
           ans = input("Do you want to delete '" + save_path + "'?")
           if ans == "y":
               for d in dirs:
                    shutil.rmtree(d)

       if Path(save_path).is_file():
           os.remove(save_path)
       Path(save_path).mkdir(exist_ok=True, parents=True)
   args = {}
   args["conf"] = Path(exp_conf).stem
   args["save_path"] = save_path
   args["load_path"] = "" 
   args["is_debug"] = debug
   if not download_model:
       args["load_path"] = mylogs.pretPath 
   args["experiment"] = "%" + experiment # % forces to reserve the value as it is  
   args["version"] = version 
   args["break_point"] = break_point 
   args["preview"] = preview 
   args["repeat"] = repeat 
   tags = exp_args["tag"] if "tag" in exp_args else ["expid"] 
   full_tags = exp_args["full_tag"] if "full_tag" in exp_args else ["expid"] 
   if break_point:
       mylogs.setbp(break_point)

   all_vars = [x.strip("--") for x in ctx.args]
   var_names = [x.split("=")[0] for x in all_vars]
   values = [x.split("=")[1].split("#") for x in all_vars]
   var_dict = {k:n for k,n in zip(var_names, values)} 
   _mvars = []
   for var in main_vars.split("--"):
       if "=" in var:
           var_name = var.split("=")[0]
           assert var_name in exp_args, var_name +" must be in experiment variables (config)"
           var_item = var.split("=")[1].split("#")
           var_dict["@" + var_name] = var_item
           _mvars.append(var_name)
   if _mvars: main_vars = _mvars
   for key,val in var_dict.items():
       multi = [item for item in val if re.match("multi-(.*)", item)]
       members = [x.strip("@") for x in val if not x in multi and not "@" in x.strip("@")]
       if multi:
           ext = []
           for m in multi:
               _, l = m.split("-")
               l = len(members) if l == "all" else int(l)
               val.remove(m)
               comb = itertools.combinations(members, l)
               ext.extend(["@".join(c) for c in comb])
           val = ext + val
           var_dict[key] = val

   var_names = list(var_dict.keys())
   values = list(var_dict.values())
   inp_exp_vars = exp_vars
   mylogs.bp("run")
   if not main_vars:
       main_vars = [vv.strip("@") for vv in var_names if vv.endswith("@")]
   if not exp_vars:
       #if main_vars:
       #    exp_vars = main_vars
       #else:
       exp_vars = [vv.strip("@") for vv in var_names if vv.startswith("@")]
   elif type(exp_vars) != list:
       exp_vars = inp_exp_vars = [exp_vars]
   if exp_vars and not log_var:
       log_var = exp_vars[0]
   full_tags.extend([x for x in exp_vars if not "^" in x])
   args["log_var"] = log_var 
   for ii, (vv, cc) in enumerate(zip(var_names, values)):
      if len(cc) > 1:
           if vv.startswith("@") or vv.endswith("@"):
               vv = vv.strip("@")
               tags.append(vv.strip("^"))
           full_tags.append(vv.strip("^"))
           values[ii] = [x for x in cc if not x.startswith("!")] 
           if (exp_vars and not vv in exp_vars) or (main_vars and not vv in main_vars):
               values[ii] = [values[ii][0]] # ignore the rest of values for this item 
      if len(values[ii]) == 1:
           if not vv.startswith("@"):
               exclude_list.append(vv)
           vv = vv.strip("@")
   var_names = [vv.strip("@") for vv in var_names]

   full_tags = list(set(full_tags))
   for pv in inp_exp_vars:
       assert pv in full_tags, f"Eror: {pv} must be 'all' or one of {full_tags} which have multiple values"

   existing_exps = glob.glob(op.join(save_path, "*.json"))
   not_conf = ["break_point","expid", "total_exp", "full_tag", "tag", "preview", "output_dir", "experiment", "trial", "num_target_prompts", "num_random_masks", "per_device_train_batch_size"]
   args["full_tag"] = full_tags 
   tot_comb = [dict(zip(var_names, comb)) for comb in itertools.product(*values)]
   ii = len(existing_exps) + 1
   exps_done = 0
   orig_args = args.copy()
   total = len(tot_comb)
   args["total_exp"] = total
   logger.info("Total experiments:%s", total)
   mylogs.bp("comb")
   old_comb = None
   ctags = []
   for comb in tot_comb:
       if old_comb is not None:
           diff_comb = DeepDiff(comb, old_comb) 
           if "values_changed" in diff_comb:
               vc = diff_comb["values_changed"]
               for item in vc:
                   val = item.replace("root['","").replace("']","")
                   if not val in ctags:
                       ctags.append(val)
       old_comb = comb.copy()

   args["tag"] = ctags 
   for comb in tot_comb:
       _output_dir = []
       prev_name = ""
       prev_item = ""
       conflict = "" 
       mvars = {}
       for kk, (var_name,var_item) in enumerate(comb.items()):
           if var_name.startswith("^") and prev_name:
               prev_vals = values[kk-1]
               cur_vals = values[kk]
               assert len(prev_vals) == len(cur_vals), "Pair variables must have same number"
               pairs = zip(prev_vals, cur_vals)
               if not (prev_item, var_item) in pairs:
                   conflict = prev_name + ":" + prev_item + " "+ var_name + ":" + var_item
                   break
           var_name = var_name.strip("^")
           args[var_name]=var_item
           if var_name in main_vars:
               mvars[var_name] = var_item
           if not var_name in exclude_list:
               _output_dir.append(var_name + "_" + str(var_item))
           prev_name = var_name
           prev_item = var_item
       if conflict:
           print(f"Dep var observed {conflict} ignored")
           continue
       ii += 1
       if max_exp > 0 and exps_done > max_exp:
           print(f"Max number of exp reached {max_exp} ")
           return

       args["expid"] = ii if not "expid" in exp_args else str(exp_args["expid"]) + "-" + str(ii)
       args["main_vars"] = mvars
       args = {**exp_args, **args}
       #_output_dir.append(str(args["expid"]))
       ee = int(args["expid"]) 
       _output_dir = str(ee)
       output_dir = os.path.join(save_path, _output_dir)
       while Path(output_dir).exists():
           ee += 1 
           _output_dir = str(ee)
           output_dir = os.path.join(save_path, _output_dir)
       args["expid"] = experiment.split("/")[-1] + "-" + str(ee)
       if not save_path:
           output_dir = os.getcwd()
       args["output_dir"] = "%" + output_dir 
       exp_conf = json.dumps(args, indent=2)
       if preview == "conf":
           print(f"================ {ii}/{total} =====================")
           print(exp_conf)
           with open("logs/exp_" + str(ii) + ".json","w") as f:
               print(exp_conf, file=f)
           continue
       # break point before running to check arguments (breakpoint must be check)
       mylogs.bp("check")
       tags_dict = mylogs.get_tag(tags, args)
       full_tags_dict = mylogs.get_tag(full_tags, args)
       #title = "@".join(list(tags_dict.values()))
       title =  mylogs.get_tag(tags, args, as_str=True)
       exp_exists = False
       if existing_exps:
           for ee in existing_exps:
               if preview == "ex-why":
                   print("Checking existaince for ", ee)
               with open(ee) as f:
                   jj = json.load(f)
                   are_equal = True
                   for k,v in args.items():
                       if not k in not_conf: 
                           if not k in jj or strval(v) != strval(jj[k]):
                               are_equal =False
                               if preview == "ex-why":
                                   print("It's not equal to because ", k, " is ",v, " against ", strval(jj[k]))
                               break
               if are_equal:
                  print(ii, " is equal to ", ee)
                  output_dir = jj["output_dir"].strip("%")
                  if glob.glob(op.join(output_dir, "*.tsv")):
                      trial = int(jj["trial"]) + 1 if "trial" in jj else 2
                      exp_exists = True
                  break
       args["trial"] = trial
       if preview == "tag":
           print(f"=#============== {ii}/{total} =====================")
           conf_str = json.dumps(full_tags_dict, indent=2)
           print(conf_str)
           if exp_exists:
               print("=============== DONE ===========")
           with open("logs/exp_" + str(ii) + ".tag","w") as f:
               print(conf_str, file=f)
           continue
       if exp_exists:
           args["output_dir"] = "%" + output_dir 
           if not preview and not repeat:
              print("Skipping experiment ", ii, ": The experiment already exists!")
              continue 
       # preview existing experiments 
       if preview == "ex" or preview == "ex-why" or preview == "exists": #
           continue
       done = "na"
       if debug:
           ctx.invoke(train, **args)
       else:
           try:
               done = ctx.invoke(train, **args)
               if done != "has_conflict" and done != "is_repeated":
                   conf_fname = os.path.join(save_path,"conf_"+str(args["expid"])+".json")
                   with open(conf_fname, "w") as f:
                       print(exp_conf, file=f)
                   exps_done += 1
               elif preview == "lict":
                   c = input("check for conflicts!")
           except Exception as e:
               print(f"================ {ii}/{total} =====================")
               exp_conf = json.dumps(args, indent=2)
               print(exp_conf)
               raise Exception("An error occured in the experiment")
       if preview == "one" or (preview == "data" and done == "data_preview"):
           return

# m3
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
    repeat = kwargs.setdefault("repeat",False)
    log_var = kwargs.setdefault("log_var","")
    main_vars = kwargs.setdefault("main_vars",{})
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
    torch.autograd.set_detect_anomaly(True)
    training_args.report_to = kwargs["report_to"] 
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


    mylogs.bp("nsp")
    num_prompts = kwargs.setdefault("num_prompts", 1) 
    target_prompt_length = adapter_args.num_prompt_tokens
    source_prompt_length = adapter_args.num_prompt_tokens
    load_source_prompts = kwargs.setdefault("load_source_prompts", True) 
    use_private_prompts = kwargs.setdefault("use_private_prompts", False)
    use_source_set = kwargs.setdefault("use_source_set", False)


    task_source_prompts_set ={}
    tasks = data_args.task_name
    for task_name in tasks:
        tid = task_name
        if not tid in task_source_prompts_set:
           task_source_prompts_set[tid] = []
        rel_sh = REL_TO_SHARED_TOKENS[task_name] if task_name in REL_TO_SHARED_TOKENS else task_name
        task_source_prompts_set[tid].extend(rel_sh.split())

    nsp = 0
    if use_source_set:
        nsp = max([len(s) for s in task_source_prompts_set.values()])
    if data_args.source_prompts is not None:
        nsp = len(data_args.source_prompts) 
    nsp += kwargs.setdefault("num_source_prompts", nsp) 
    num_source_prompts = nsp 
    num_target_prompts = 1
    if model_args.attn_tuning is True:
        num_target_prompts = kwargs.setdefault("num_target_prompts",num_source_prompts) 
        ntp = num_target_prompts
        if ntp < 0: 
            num_target_prompts = num_source_prompts
        if num_source_prompts > 0:
            num_target_prompts = min(num_target_prompts, num_source_prompts)
        else:
            num_target_prompts = 1
        if model_args.attend_target and ntp < 0:
            num_target_prompts += 1
        if model_args.attend_input and ntp < 0:
            num_target_prompts += 1
        if use_private_prompts and ntp < 0:
            num_target_prompts += 1
        num_target_prompts = max(num_target_prompts, 1)
        if model_args.compose_method == "cat":
            target_prompt_length = num_target_prompts * adapter_args.num_prompt_tokens
        elif model_args.compose_method == "wcat":
            target_prompt_length = 2 * adapter_args.num_prompt_tokens
        elif model_args.compose_method == "wavg":
            pass
            #target_prompt_length = num_target_prompts * adapter_args.num_prompt_tokens
            #adapter_args.num_prompt_tokens = target_prompt_length

        kwargs["num_target_prompts"] = num_target_prompts
        mylogs.main_args["num_target_prompts"] = num_target_prompts

    kwargs["num_prompt_tokens"] = target_prompt_length 
    kwargs["source_prompt_length"] = source_prompt_length 
    kwargs["target_prompt_length"] = target_prompt_length 
    task_args = {}
    task_args["data_seed"] = data_args.data_seed
    task_args["map_labels"] = kwargs.setdefault("map_labels", True)
    task_args["train_samples"] = data_args.max_train_samples
    task_args["val_samples"] = data_args.max_val_samples
    task_args["test_samples"] = data_args.max_test_samples
    task_args["num_prompts"] = num_prompts 
    task_args["target_prompt_length"] = target_prompt_length 
    task_args["prompt_length"] = kwargs.setdefault("prompt_length", 
                                    adapter_args.num_prompt_tokens)
    task_args["fixed_length_prompt"] = adapter_args.fixed_length_prompt
    task_args["template"] = data_args.template
    task_args["add_prefix"] = data_args.add_prefix
    task_args["data_path"] = data_args.data_path
    task_args["rels"] = kwargs.rels
    task_args["task_comb"] = kwargs.task_comb
    task_args["id"] = kwargs["expid"]

    # an option to explicitly specify the method of training 
    # (pt: prompt-tuning, ft:fine-tuning, px:prefix-tuning etc.)
    method = kwargs.setdefault("method", "")
    ds_confs = kwargs.setdefault("ds_config", ["en"])
    n_tasks = len(data_args.task_name)
    _confs =["en"] * n_tasks
    for i,c in enumerate(ds_confs):
        _confs[i] = ds_confs[i]
    #if kwargs.setdefault("adjust_epochs", False):
    #    training_args.num_train_epochs *= n_tasks

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
            shutil.rmtree(training_args.output_dir)
            return "has_conflict"

    if main_vars:
        x = main_vars
        y = mylogs.prev_main_vars
        repeated_items = {k: x[k] for k in x if k in y and x[k] in y[k]}
        if len(repeated_items) == len(main_vars):
            shutil.rmtree(training_args.output_dir)
            return "is_repeated"
        for k,v in main_vars.items():
            if not k in mylogs.prev_main_vars:
                mylogs.prev_main_vars[k] = []
            mylogs.prev_main_vars[k].append(v)
        if preview == "mavar":
            return 

    if log_var:
       mylogs.plog.handlers.clear()
       mylogs.add_handler(mylogs.plog, log_var + "_" + str(kwargs[log_var]))
       mylogs.plog.info(exp_conf)
    ###### Collect experiment infos
    exp_info = {}
    for k,v in kwargs.items():
        if not k in exp_info:
            exp_info[k] = v


    wandb_dir = kwargs.save_path #op.join("logs", experiment)
    Path(wandb_dir).mkdir(parents=True, exist_ok=True)
    experiment = kwargs.experiment
    tags_dict = mylogs.get_tag(tag, kwargs)
    if not preview or preview=="one":
       wandb.init(
          # Set the project where this run will be logged
          project= experiment.replace("#","-").replace("/","-")[:100], 
          name=title,
          dir=wandb_dir,
          settings=wandb.Settings(symlink=False),
          # Track hyperparameters and run metadata
          config=tags_dict
       )
    if wandb.run is not None:
        exp_info["runid"] = wandb.run.id
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
            #existing_results = glob.glob(op.join(training_args.output_dir, "*.tsv"))
            #if existing_results and not preview and not repeat:
            #    print("Skipping experiment:", training_args.output_dir)
            #    return "skipped" 
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
        #handlers=[logging.StreamHandler(sys.stdout)],
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
    tasks = data_args.task_name
    mylogs.bp("steps")
    total_samples = 0
    for ti, task_name in enumerate(tasks, start=1):
         t_args = dotdict(task_args.copy())
         task = AutoTask.get(task_name, None, task_args=t_args)
         total_samples += data_args.max_train_samples * task.samples_per_head

    training_args.per_device_train_batch_size = min(total_samples, training_args.per_device_train_batch_size)
    steps = 0
    if training_args.do_train:
        steps = total_samples * training_args.num_train_epochs // (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size)
    mylogs.bp("steps")
    if model_args.anneal_rate is None: 
        anneal_rate = 1/(steps + 5) 
    else:
        anneal_rate = model_args.anneal_rate
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

    mylogs.bp("config")
    config.train_task_adapters = adapter_args.train_task_adapters
    config.prefix_tuning = adapter_args.prefix_tuning
    config.prompt_tuning = adapter_args.prompt_tuning #my option
    config.attn_tuning = model_args.attn_tuning
    config.attn_method = model_args.attn_method
    config.compose_method = model_args.compose_method #my option
    config.select_method = model_args.select_method #my option
    config.target_share_temperature = model_args.target_share_temperature
    config.anneal_min = model_args.anneal_min # my option
    config.anneal_dir = model_args.anneal_dir # my option
    config.anneal_rate = anneal_rate # my option
    config.attend_target = model_args.attend_target
    config.num_target_prompts = num_target_prompts
    config.attend_private = use_private_prompts 
    config.source_prompts_order = kwargs.setdefault("source_prompts_order", "desc")
    config.sel_positives = kwargs.setdefault("sel_positives", False)
    config.attend_for = kwargs.setdefault("attend_for", "inp_target")
    config.attend_source = model_args.attend_source #my option
    config.attend_input = model_args.attend_input #my option
    config.route_method = model_args.route_method #my option
    config.normalize = kwargs.setdefault("normalize", True)
    config.add_target = model_args.add_target #my option
    config.target_share = model_args.target_share #my option
    config.sig_coef = model_args.sig_coef #my option
    config.apply_softmax_to = kwargs.setdefault("apply_softmax_to", "all") #my option
    config.shared_attn = model_args.shared_attn
    if model_args.prompt_embedding_path:
        config.prefix_num = len(model_args.prompt_embedding_path) 
    else:
        config.prefix_num = model_args.prefix_num
    config.num_target = len(data_args.task_name)
    config.temperature = model_args.temperature
    config.learned_temperature = model_args.learned_temperature
    config.learn_attention = model_args.learn_attention
    config.learn_source_prompts = model_args.learn_source_prompts
    config.learn_target_prompts = model_args.learn_target_prompts
    adapter_config = get_adapter_config(
        adapter_args, data_args, training_args, config)

    # Set tokenizer
    if "mt5" in model_name_or_path:
        tokenizer = MT5TokenizerFast.from_pretrained(model_name_or_path)
    elif "pars" in model_name_or_path:
        tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path)
    else:
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

    prompts_dir = model_args.prompt_encoders_dir
    if prompts_dir and not prompts_dir.startswith("/") and not prompts_dir == "save_path":
        prompts_dir = op.join(mylogs.pretPath, prompts_dir) 
    else:
        base_folder = Path(kwargs.save_path)
        base_folder_stem = base_folder.stem
        base_folder_name = base_folder.name
        prompts_dir = training_args.output_dir.replace(base_folder_name, base_folder_stem)

    router_prefix = kwargs.setdefault("router_prefix", "") 
    if not router_prefix or router_prefix == "1":
        router_prefix = str(data_args.max_train_samples)

    router_prefix = "-".join(sorted(data_args.task_name)) + "-" + str(num_source_prompts)
    dpath = os.path.join(prompts_dir, router_prefix + "_router.pt")
    mylogs.bp("router")
    if model_args.attn_tuning is True:
       if Path(dpath).is_file():
          model.update_router(dpath)
          mask = model.encoder.router[model.encoder.router > 0.1] 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ######################## My code pppppp
    mylogs.bp("penc")
    prompts_prefix = kwargs.setdefault("prompts_prefix", "") 
    prompts_prefix = str(prompts_prefix)
    if prompts_prefix is None: prompts_prefix = ""
    #prompts_prefix = prompts_prefix + "_" + str(data_args.template)
    if not prompts_prefix or prompts_prefix == "1":
        prompts_prefix = str(data_args.max_train_samples)
    if not load_source_prompts and model_args.attn_tuning:
        prompts_prefix = prompts_prefix + "_" \
                + kwargs.experiment.split("/")[0] \
                + "_" + kwargs.expid

    if not router_prefix:
        router_prefix = prompts_prefix

    if adapter_args.prompt_tuning:
        added = add_specials(tokenizer)
        logger.info("%s tokens was addded", added)
        model.resize_token_embeddings(len(tokenizer))
        # mmmmmmmmmmmmm Add target prompts
        mylogs.bp("encoders")
        prompts = {}
        prompt_sharing = kwargs.setdefault("prompt_sharing", "shared_encoders") 
        tasks = data_args.task_name
        n_tasks = len(tasks)
        task_prompts = {}
        task_source_prompts_set ={}
        #  tttttttttttt
        for ti, task_name in enumerate(tasks, start=1):
             task_args["id"] = ti
             t_args = dotdict(task_args.copy())
             task = AutoTask.get(task_name, None, task_args=t_args)
             p = task.get_prompts()
             prompts = {**prompts, **p}
             tid = task_name #get_id()
             if not tid in task_prompts:
                 task_prompts[tid] = []
                 task_source_prompts_set[tid] = []
             for k,v in p.items():
                 task_prompts[tid].extend(v)
             rel_sh = REL_TO_SHARED_TOKENS[task_name] if task_name in REL_TO_SHARED_TOKENS else task_name
             task_source_prompts_set[tid].extend(rel_sh.split())

        for name, prompt_tokens in prompts.items():
            extend_tokenizer(tokenizer, prompt_tokens)

        # mmmmmmmmmmmmm Add source prompts
        prompt_encoders = []
        source_prompts = []
        nsp = kwargs.setdefault("num_source_prompts", nsp) 
        if data_args.source_prompts:
            source_prompts = ["source_" + sp for sp in data_args.source_prompts]
        if nsp > 0:
            source_prompts.extend(
                    ["source_com" + str(sp) for sp in range(nsp)])
        if use_private_prompts:
            source_prompts.extend(["source_for_" + t for t in data_args.task_name])
        if use_source_set:
            pset = []
            for t in data_args.task_name:
                pset.extend(task_source_prompts_set[t])
            pset = set(pset)
            source_prompts.extend(["source_" + t for t in pset]) 

        kwargs["num_source_prompts"] = len(source_prompts)
        mylogs.main_args["num_source_prompts"] = len(source_prompts)
        for prompt in source_prompts: 
            encoder, enc_type = create_encoder(prompt, model, tokenizer, 
                    prompt_tokens=[],
                    is_source = True,
                    length = adapter_args.num_prompt_tokens,
                    encoder_type=adapter_args.prompt_encoder_type) 
            if "_for" in encoder.name:
                encoder.is_shared = False
                encoder.is_private = True
            if kwargs.setdefault("init_from_words", False):
                encoder.init_embs_from_words(model.get_input_embeddings())
            if load_source_prompts and not "_for" in prompt: 
                # and not "_com" in prompt and not "_for" in prompt:
                ignore_if_not_exist = kwargs.setdefault("ignore_if_not_exist", False)
                if bp == "load":
                    breakpoint()
                is_loaded = encoder.load(prompts_dir, 
                        prefix=prompts_prefix,
                        ignore_if_not_exist=ignore_if_not_exist,
                        length = adapter_args.num_prompt_tokens)
                if is_loaded:
                    logger.info("%s was loaded", encoder.name)
                else:
                    logger.info("% doesn't exist and wasn't loaded", encoder.name)
                if bp == "load":
                    breakpoint()
                exp_info["load_" + prompt] = is_loaded
            prompt_encoders.append(encoder)

        ############################ Create Target Prompt Encoders #############
        encoders_prompts = prompts
        # task prompts has one encoder per task where they could have shared tokens
        # shared encoders has one encoder per prompt ids. 
        # If two tasks use similar prompts they recieve the output of same encoders
        if prompt_sharing == "shared_prompts":
            encoders_prompts = task_prompts
        model.resize_token_embeddings(len(tokenizer))
        load_prompts = kwargs.setdefault("load_prompts", False) 
        attend_to_all = kwargs.setdefault("attend_to_all", False) 
        target_prompts=[n for n,p in encoders_prompts.items() if p[0].startswith("<tar-")]  
        # create and load target prompts
        mylogs.bp("mask")
        num_attend_to = len(source_prompts) + len(target_prompts) + 1 # one for input 
        for name, prompt_tokens in encoders_prompts.items():
            encoder, enc_type = create_encoder(name, model, tokenizer, 
                    prompt_tokens, 
                    encoder_type=adapter_args.prompt_encoder_type) 
            if name in task_source_prompts_set:
                encoder.attend_to.extend(
                        ["source_" + x for x in task_source_prompts_set[name]])
            if prompt_tokens[0].startswith("<tar-"):
                encoder.is_target = True
                nn = name.replace("tar-","")
                encoder.attend_to.extend(["source_for_" +  nn])
            encoder.attend_to_mask = [1]*num_attend_to 
            attn_flag = False
            for i, n in enumerate(source_prompts, start=1):
                encoder.attend_to_mask[i] = 0 
                if n in encoder.attend_to:
                    encoder.attend_to_mask[i] = 1 
                    attn_flag = True
                if "_com" in n or attend_to_all:
                    encoder.attend_to_mask[i] = 1 
                    attn_flag = True
                if "_for" in n and not n in encoder.attend_to:
                    encoder.attend_to_mask[i] = 0 
                    attn_flag = True
            if not attn_flag or (not use_private_prompts and not use_source_set): 
                encoder.attend_to_mask = [1]*num_attend_to # attend to all 
            if kwargs.setdefault("init_from_words", False):
                encoder.init_embs_from_words(model.get_input_embeddings())
            if not model_args.attn_tuning and  load_prompts: 
                ignore_if_not_exist = kwargs.setdefault("ignore_if_not_exist", False)
                # if not model_args.attn_tuning or encoder.is_source:
                is_loaded = encoder.load(prompts_dir, 
                        prefix=prompts_prefix,
                        ignore_if_not_exist=ignore_if_not_exist,
                        as_saved=True,
                        length = target_prompt_length)
                ignore_train_if_exist = kwargs.setdefault("ignore_train_if_exist", False)
                if is_loaded and ignore_train_if_exist:
                    training_args.do_train = False
                    logger.info("%s training was ignored", encoder.name)
                if bp == "load":
                    breakpoint()
            prompt_encoders.append(encoder)

        exp_info["num_encoders"] = len(prompt_encoders)
        exp_info["len_encoders"] = ",".join([str(e.length) for e in prompt_encoders])
        exp_info["taginfo"].append("len_encoders")
        model.encoder.set_encoders(prompt_encoders, 
            source_prompts, 
            source_prompt_length,
            target_prompt_length) 
        model.resize_token_embeddings(len(tokenizer))

    if log_var and preview == "encoders":
        mylogs.plog.info("======== Number of encoders: %s", len(prompt_encoders))
        for ii, e in enumerate(prompt_encoders):
            mylogs.plog.info("%s) Name:%s, length: %s", ii, e.name, e.length)
            mylogs.plog.info("Tokens:%s", e.prompt_tokens)
            mylogs.plog.info("Ids:%s ", e.prompt_ids)
            mylogs.plog.info(e)
        return 

    mylogs.bp("freeze")
    mylogs.bp("rgrad")
    rgrad = len([p for p in model.parameters() if p.requires_grad])
    nrgrad = len([p for p in model.parameters() if not p.requires_grad])
    mylogs.plog.info("Before freeze: requires grad: %s   Not requires grad: %s", rgrad, nrgrad)
    model = modify_model_after_init(
        model, training_args, adapter_args, adapter_config)
   
    learn_loaded_prompts = kwargs.setdefault("learn_loaded_prompts", False) 
    learn_private_prompts = kwargs.setdefault("learn_private_prompts", True) 
    if adapter_args.prompt_tuning:
        for encoder in prompt_encoders: 
            if encoder.is_source:
                if model_args.learn_source_prompts:
                    if encoder.is_loaded and not learn_loaded_prompts:
                        continue
                    if encoder.is_private and not learn_private_prompts:
                        continue
                    for n,p in encoder.named_parameters():
                        p.requires_grad = True
            else:
                if model_args.learn_target_prompts:
                    for n,p in encoder.named_parameters():
                        p.requires_grad = True

    rgrad = len([p for p in model.parameters() if p.requires_grad])
    nrgrad = len([p for p in model.parameters() if not p.requires_grad])
    exp_info["rgrad-nrgrad"] = str(rgrad) + "|" + str(nrgrad)
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
    hit_count = kwargs.setdefault("hc", 3)
    def preprocess_function(examples, max_target_length, task_id=None):
        model_inputs = tokenizer(examples['source'], max_length=data_args.max_source_length,
                                 padding=padding, truncation=True)
        if preview == "data":
            mylogs.plog.info("sourece: %s", examples["source"][:hit_count])
            mylogs.plog.info("target: %s", examples["target"][:hit_count])

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
        mylogs.bp("train_test_data")
        model_inputs["extra_fields"] = examples['extra_fields']  
        if task_id is not None:
            model_inputs["task_ids"] = [
                task_id for _ in examples['extra_fields']]
        return model_inputs

    column_names = ['source', 'target', 'extra_fields']
    performance_metrics = {}
    task_args = dotdict(task_args.copy())
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
        if trainer_shuffle:
            train_dataset = concatenate_datasets(train_datasets)
        else:
            mylogs.bp("myint")
            train_dataset = my_interleave_datasets(train_datasets, 
                batch_size=training_args.per_device_train_batch_size)
    if preview == "data":
       return "data_preview" 
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
        data_info["train"] = train_dataset['extra_fields'] if training_args.do_train else None

    def compute_metrics(eval_preds):
        preds, labels, data_info, task = eval_preds
        post_processor = AutoPostProcessor.get(task, tokenizer,
                                               data_args.ignore_pad_token_for_loss)
        decoded_preds, decoded_labels = post_processor.process(
            preds, labels, data_info)
        task = AutoTask.get(task, None, task_args=task_args)
        mylogs.bp("compute")
        decoded_preds, decoded_labels = task.post_process(decoded_preds, decoded_labels)
        result = {}
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result

    # If you want to use a different learning rate for attention layer, initialize an optimizer using the learning rate here.
    grouped_params = []
    all_parameters = set([p for p in model.parameters() if p.requires_grad])
    attn_params = []
    prompt_params = []
    if model_args.attn_learning_rate is not None:
        for name, param in model.named_parameters():
            if (name == "encoder.attn_W_up.weight" 
                or name == "encoder.attn_W_down.weight" 
                or name == "encoder.layer_norm.weight"):
                attn_params.append(param)
            if name == "encoder.router" or name == "encoder.target_router":
                attn_params.append(param)

        attn_params = set(attn_params)
        grouped_params.append({'params': list(attn_params), 
            'lr': model_args.attn_learning_rate})
        

    ########### My Code
    prompt_learning_rate = model_args.prompt_learning_rate 
    target_prompt_learning_rate = model_args.target_prompt_learning_rate 
    source_prompt_learning_rate = model_args.source_prompt_learning_rate 
    if source_prompt_learning_rate is None:
        source_prompt_learning_rate = prompt_learning_rate 
    if target_prompt_learning_rate is None:
        target_prompt_learning_rate = prompt_learning_rate 
    src_prompt_params = []
    tgt_prompt_params = []
    mylogs.bp("opt")
    learning_rate = training_args.learning_rate
    if adapter_args.prompt_tuning:
        learning_rate = target_prompt_learning_rate
        for encoder in model.prompt_encoders:
           para_list =[p for p in encoder.parameters() if p.requires_grad]
           if para_list: 
               if encoder.is_source and not encoder.is_private:
                   src_prompt_params.extend(para_list)
               else:
                   tgt_prompt_params.extend(para_list)

        src_prompt_params = set(src_prompt_params)
        tgt_prompt_params = set(tgt_prompt_params)
        grouped_params.append({'params': list(src_prompt_params), 
            'lr': source_prompt_learning_rate})
        grouped_params.append({'params': list(tgt_prompt_params), 
            'lr': target_prompt_learning_rate})
        prompt_params = list(src_prompt_params) + list(tgt_prompt_params)

    other_params = all_parameters - set(attn_params) - set(prompt_params)
    other_params = list(other_params)
    if other_params:
        grouped_params.append({'params': other_params, 'lr': training_args.learning_rate})
    #### ooooo 
    mylogs.bp("opt")
    if kwargs.opt_type == "sep":
        optim, scheduler = get_optimizer(model, steps,
                source_prompt_learning_rate, 
                model_args.attn_learning_rate, 0.01)
    else:
        optim = AdamW(grouped_params, lr=learning_rate)
        if training_args.warmup_steps is not None:
            warmup_steps = training_args.warmup_steps
        else:
            warmup_steps = 0.2 * steps
        scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=warmup_steps, 
            num_training_steps=steps)
    name = data_args.dataset_name[0] 
    task_metric = TASK_TO_METRICS[name] if name in TASK_TO_METRICS else ["rouge"]
    if training_args.do_eval: 
        eval_ds = my_interleave_datasets(list(eval_datasets.values()), batch_size=2)
    else: 
        eval_ds = None
    wb_callback = WBCallback()
    anneal_callback = AnnealCallback() 
    ptlr_callback = PTLearningRateCallback()
    callbacks = []
    if adapter_args.prompt_tuning:
       callbacks = [ptlr_callback, wb_callback, anneal_callback]
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
            callbacks = callbacks, 
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
            callbacks = callbacks, 
            shuffle = trainer_shuffle,
            save_checkpoint = kwargs.setdefault("save_checkpoint", False),
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            evaluation_metrics=task_metric,
            multi_task_compute_metrics=compute_metrics_fn,
            shared=model_args.shared_attn)

    # Exit program if user wants to check some settings 
    if preview and preview != "one":
        return
    # Saves training config.
    if trainer.is_world_process_zero():
        os.makedirs(training_args.output_dir, exist_ok=True)
        save_training_config(config_file, training_args.output_dir)


    def load_model(load_path, lsp=False):
        #model.load_encoders(load_path, load_source_prompts=lsp)
        dpath = os.path.join(load_path, "attn_W_down.pt")
        attention_paths = [dpath, 
                os.path.join(load_path, "attn_W_up.pt")]
        if model_args.attn_tuning is True and Path(dpath).is_file():
            trainer.model.update_attention_weights_sub(attention_paths)
            if model_args.load_layer_norm and "layer_norm_bias.pt" in load_path: 
                trainer.model.update_layer_norm_weights(load_path)
        dpath = os.path.join(load_path, router_prefix + "_router.pt")
        if model_args.attn_tuning is True:
            if Path(dpath).is_file():
                trainer.model.update_router(dpath)
            else:
                dpath = os.path.join(prompts_dir, router_prefix + "_router.pt")
                if Path(dpath).is_file():
                    trainer.model.update_router(dpath)
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

        print("=================== Training ====================")
        print("Experiment: ", mylogs.args("expid"), "/", mylogs.args("total_exp"))
        print("Tags: ", mylogs.get_tag(as_str=True))
        print("=================================================")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        if training_args.compute_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"total_time in minutes ": total_time})

        # Load best model
        if trainer.best_prompt_checkpoint is not None:
            best_chk_path = trainer.best_prompt_checkpoint
            lsp = kwargs.setdefault("load_source_prompts", False)
            load_model(best_chk_path, lsp=lsp)

        # Save prompts
        if adapter_args.prompt_tuning:
            #if not model_args.attn_tuning: 
            #    prompts_prefix = "pt_" + prompts_prefix 
            #else: 
            #    prompts_prefix = "att_" + prompts_prefix 
            #prompts_prefix = prompts_prefix.strip("_")
            ssp = kwargs.setdefault("save_source_prompts", False) 
            model.store_encoders(output_dir = training_args.output_dir,
                                 save_source_prompts = ssp, 
                                 prefix=prompts_prefix,
                                 router_prefix=router_prefix)

            prompts_to_save = kwargs.setdefault("save_these_prompts", []) 
            if prompts_to_save:
                Path(prompts_dir).mkdir(parents = True, exist_ok=True)
                model.store_encoders(output_dir = prompts_dir, 
                        prompts_and_router_only=True, 
                        prompts_to_save = prompts_to_save, 
                        save_source_prompts = ssp,
                        prefix=prompts_prefix, 
                        router_prefix=router_prefix)

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
            lsp = kwargs.setdefault("load_source_prompts",False)
            load_model(training_args.output_dir, lsp=lsp)

        if  model_args.shared_attn is False:
            for task, eval_dataset in eval_datasets.items():
                metrics = trainer.evaluate(eval_dataset=eval_dataset,
                                           max_length=data_args.val_max_target_length, 
                                           num_beams=data_args.num_beams,
                                           )
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)
                metric_to_check = training_args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]
                wandb.run.summary[f"evaluation_{metric_to_check}"] = metric_value 

    # Test
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
            test_datasets = {test_dataset + "_" + test_dataset_config: AutoTask.get(test_dataset, test_dataset_config,
                                                        task_args=task_args).get(
                split="test",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_test_samples, lang=data_args.lang_name, file_name=data_args.test_file)
                for test_dataset, test_dataset_config in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)}
            mylogs.bp("test_dataset")

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

        if has_extra:
            data_info["test"] = test_datasets[data_args.test_dataset_name[0] + "_" + data_args.test_dataset_config_name[0]]['extra_fields'] if training_args.do_test else None
        logger.info("*** Test ***")
        # multi-task evaluations
        def evaluate_test(task, test_dataset, save_to, ds_name, gen_conf = {}):
            mylogs.bp("ttt")
            predictions, labels, metrics = trainer.predict(
                    gen_conf = gen_conf,
                    test_dataset=test_dataset,
                    max_length=data_args.test_max_target_length, 
                    num_beams=data_args.num_beams,
                    metric_key_prefix="test", task=task)
            
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

            # sssssssssss
            #predictions = np.argmax(predictions, axis=1)
            #predictions = tokenizer.batch_decode(predictions)
            df = test_dataset.to_pandas()
            if bp == "test": breakpoint()
            df["pred_text1"] = ""
            df["prefix"] = ds_name
            df["template"] = data_args.template
            df["resp"] = ""
            df["time"] = mylogs.now 
            df["date"] = mylogs.today 
            df["query"] = ""
            df["langs"] = "en2en"
            for k,v in metrics.items():
                df[k] = v
            #df["rouge_score"] = 0.0
            #df["bert_score"] = 0.0
            for key, info in exp_info.items():
                if type(info) == list:
                    info = "@".join([str(inf) for inf in info])
                if type(info) == dict:
                    info = json.dumps(info)
                    info = info.replace("\n", "@")
                df[key] = info
            rouge_scorer = Rouge()
            for i, row in df.iterrows():
                mylogs.bp("=testloop") 
                extra = row["extra_fields"]
                if "event" in extra:
                    inp = extra["event"]
                else:
                    inp = tokenizer.decode(row["input_ids"], 
                        skip_special_tokens=kwargs.setdefault("skip_spcials", True)) 
                inp = re.sub(r'<.*?>','', inp)
                inp = inp.strip()
                df.at[i, "input_text"] = inp #extra["event"] 
                label = extra["tail"] if "tail" in extra else "na"
                #label = tokenizer.decode(row["labels"], 
                #skip_special_tokens=kwargs.setdefault("skip_spcials", True)) 
                label = re.sub(r'<.*?>','', label)
                label = label.strip()
                df.at[i, "target_text"] = extra["target_text"] #label 
                sel = False
                if "sel" in extra:
                    sel = extra["sel"] 
                df.at[i, "sel"] = sel 
                df.at[i, "query"] = extra["query"]  
                df.at[i, "resp"] = label # extra["resp"]  
                mylogs.bp("decode")
                pred = tokenizer.decode(predictions[i], 
                        skip_special_tokens=kwargs.setdefault("skip_spcials", True)) 
                pred = re.sub(r'<.*?>','',pred)
                pred = pred.strip()
                df.at[i, "pred_text1"] = pred
            df = df.drop(columns=["input_ids","labels","attention_mask"])
            scores = do_score(df, "rouge@bert", save_to)
            return df, scores



        ##################
        if not training_args.do_train:
            load_model(training_args.output_dir, lsp=False)
        results = {}
        gen_conf = {}
        ds_backup = None
        grm = kwargs.setdefault("gen_route_methods",["rb"])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        combs = {}
        num_random_masks = kwargs.setdefault("num_random_masks",0)
        ntp = kwargs.num_target_prompts
        mylogs.bp("ttt")
        for rm in range(num_random_masks):
            mask = model.encoder.random_attn_mask(rm, ntp)
            combs["rand-" + str(rm)] = mask
        combs["train"] = None
        ii = 0
        kk = 0
        sdf_rows = []
        if not adapter_args.prompt_tuning:
            for idx, (task, test_dataset) in enumerate(test_datasets.items()):
                task = task.split("_")[0]
                ds_conf = data_args.test_dataset_config_name[idx]
                ds_name = data_args.test_dataset_name[idx]
                ds_name = "none" if not ds_name else ds_name
                ds_conf = "none" if not ds_conf else ds_conf
                is_train = "train" if training_args.do_train else "eval"
                save_to = os.path.join(training_args.output_dir, 
                     ds_conf + "_results_" + is_train + "_" + ds_name + \
                     str(kwargs.trial) + "_" + mylogs.now + "_1.tsv")
                df, scores = evaluate_test(task, test_dataset, save_to, ds_name)
        else:
            for rm, mask in combs.items():
                img_list = []
                for route_method in grm: 
                    attend_num =len(model.encoder.prompt_encoders) + 1 # one for input
                    model.encoder.attn_scores = torch.zeros(
                        (attend_num, attend_num), device=device) 
                    for idx, (task, test_dataset) in enumerate(test_datasets.items()):
                        gen_conf["route_method"] = route_method
                        if mask is not None: 
                           gen_conf["attn_mask"] = mask 
                        else:
                           gen_conf["attn_mask"] = model.encoder.attn_mask_orig 
                        exp_info["gen_route_methods"] = route_method
                        exp_info["gen_mask"] = rm 
                        mylogs.bp("test")
                        task = task.split("_")[0]
                        ds_conf = data_args.test_dataset_config_name[idx]
                        ds_name = data_args.test_dataset_name[idx]
                        ds_name = "none" if not ds_name else ds_name
                        ds_conf = "none" if not ds_conf else ds_conf
                        is_train = "train" if training_args.do_train else "eval"
                        save_to = os.path.join(training_args.output_dir, 
                                ds_conf + "_results_" + is_train + "_" + ds_name + \
                                        "_" + route_method + "_" + str(kwargs.trial) + \
                                        "_" + mylogs.now + "_" + str(ii)  + ".tsv")

                        df, scores = evaluate_test(task, test_dataset, 
                                save_to, ds_name, gen_conf)
                        df["src_path"] = op.join(mylogs.home, data_args.data_path, 
                                                ds_conf,"test.tsv")
                        mylogs.bp("test")
                        test_rouge = wandb.run.summary["test_rouge"]
                        test_bert = wandb.run.summary["test_bert"]
                        num_preds = wandb.run.summary["num_preds"]
                        da = {}
                        da["task"] = task
                        da["route_method"] = route_method
                        da["test_rouge"] = test_rouge
                        da["test_bert"] = test_bert
                        da["num_preds"] = num_preds
                        sdf_rows.append(da)

                        ii += 1

                    mylogs.bp("pic")
                    targets = model.encoder.target_encoders_idx
                    ss1 = model.encoder.attn_scores.index_select(0, targets)
                    ssq = torch.round(ss1*100)/100
                    ss2 = model.encoder.router.index_select(0, targets)
                    mask = model.encoder.attn_mask if mask is None else mask
                    ss3 = mask.index_select(0, targets)
                    y_labels = [model.encoder.prompt_names[i] for i in targets]
                    _main_vars = main_vars.copy()
                    if "task_name" in _main_vars:
                        del _main_vars["task_name"]
                    for score in [ss1]: #, # ss2, ss3]:
                        img_buf = WBCallback.save_image(score=score, 
                            y_labels=y_labels,
                            x_labels=model.encoder.prompt_names, 
                            title = str(kwargs.expid) + str(_main_vars)  \
                                    + route_method \
                                    + "_" + model_args.compose_method \
                                    + "_" + kwargs.apply_softmax_to \
                                    + "_" + model_args.attn_method) 
                        if img_buf:
                            im = Image.open(img_buf)
                            img_list.append(im)

                if img_list:
                    new_im = combine_y(img_list)
                    fname = "pred_" + str(exp_info["expid"]) + "_" + rm + "_" + route_method 
                    wandb.log({fname:wandb.Image(new_im)})

            mylogs.bp("diff")
            sdf = pd.DataFrame(data=sdf_rows)
            da = {}
            targets = model.encoder.target_encoders_idx
            ss1 = model.encoder.attn_scores.index_select(0, targets)
            ss2 = model.encoder.router.index_select(0, targets)
            _tag = kwargs.setdefault("tag",[])
            da = mylogs.get_tag(_tag)  
            #if diff_args:
            #    for k,v in diff_args["values_changed"].items():
            #        if not "output_dir" in k and not "expid" in k:

            #           da[k] = v
            _main_vars = main_vars.copy()
            if "task_name" in _main_vars:
                del _main_vars["task_name"]

            for score in [ss1, ss2]:
                img_buf = WBCallback.save_image(score=score, 
                   y_labels=y_labels,
                   x_labels=model.encoder.prompt_names, 
                   title = str(kwargs.expid) + str(_main_vars) \
                            + model_args.compose_method \
                            + "_" + kwargs.apply_softmax_to \
                            + "_" + model_args.attn_method,
                    df=None) 
                if img_buf:
                    cur_img = Image.open(img_buf)
                    #tags_img = tag_to_image(da, get_image=True)
                    #cur_img = combine_x([tags_img, cur_img])
                    sp = op.join(kwargs.save_path, "images") 
                    Path(sp).mkdir(exist_ok=True, parents=True)
                    pic = "router_" + str(exp_info["expid"])
                    pp = sp + "/pred_" + pic + ".png"
                    if Path(pp).is_file():
                        _image = Image.open(pp)
                        cur_img = combine_y([cur_img, _image])
                    cur_img.save(pp)
                kk += 1

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
            if adapter_args.prefix_tuning:
                save_prompts(model, output_dir=new_dir, 
                     prefix_dir = prefix_dir,
                     attn_tuning=model_args.attn_tuning,
                     shared_attn=model_args.shared_attn, num_target=config.num_target, task_name=data_args.task_name)
            if adapter_args.prompt_tuning:
                save_prompts_flag = kwargs.setdefault("save_prompts", False) 
                if save_prompts_flag:
                    Path(op.join(new_dir, "prompts")).mkdir(parents = True, exist_ok=True)
                    model.store_encoders(output_dir = prompts_dir, prompts_only=True)

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
            if (model_args.load_layer_norm is True 
                and "layer_norm_bias.pt" in checkpoint_dir):
                trainer.model.update_layer_norm_weights(checkpoint_dir)
            dpath = os.path.join(checkpoint_dir, router_prefix + "_router.pt")
            if model_args.attn_tuning is True and Path(dpath).is_file():
                trainer.model.update_router(dpath)
            else:
                dpath = os.path.join(prompts_dir, router_prefix + "_router.pt")
                if Path(dpath).is_file():
                    trainer.model.update_router(dpath)

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
    wandb.finish()
    return results

if __name__ == "__main__":
   cli()
