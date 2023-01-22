from collections import OrderedDict
from datasets import Dataset
import collections
import abc
import functools
from typing import Callable, List, Mapping
from utils import pad_punctuation
from metrics import metrics
from .utils import round_stsb_target, defdict
import datasets
import logging
import numpy as np
import torch
import re
from attempt.maps import *
import attempt.mylogs as mylogs
from itertools import cycle, islice

logger = logging.getLogger(__name__)

class AbstractTask(abc.ABC):
    name = NotImplemented
    do_shuffle = True # My code
    config = NotImplemented
    prefix = NotImplemented
    preprocessor: Callable = NotImplemented
    metric = NotImplemented
    metric_names = NotImplemented
    split_map = None
    labels_list = None
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}
    small_datasets_without_all_splits = ["cola", "wnli", "rte", "superglue-cb", "superglue-copa", "superglue-multirc",
                                         "superglue-wic", "superglue-wsc.fixed", "superglue-rte", "mrpc", "stsb",
                                         "superglue-boolq", "xsum", "scitail"]
    large_data_without_all_splits = ["qqp", "qnli", "superglue-record", "sst2", "squad", "snli", "anli",
                                     "amazon_polarity", "yelp_polarity", "winogrande", "newsqa", "searchqa", "triviaqa", "nq", "hotpotqa"]

    def __init__(self, config, task_args):
        self.config = config
        self.seed = task_args.data_seed
        self.template = task_args.template
        ## list of prompts
        self.prompt_set = {} 
        self.prompt_length = task_args.num_prompt_tokens
        self.common_length = task_args.num_common_tokens
        self.task_args = task_args
        self.counter = {} #counter for logging items

    def get_max_target_length(self, tokenizer, default_max_length):
        if self.labels_list is not None:
            return max([len(tokenizer.encode(label)) for label in self.labels_list])
        return default_max_length


    def check_n_obs(self, n_obs, total_size):
        if n_obs < 0 or (n_obs is not None and n_obs > total_size):
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def shuffled_indices(self, dataset):
        if not self.do_shuffle:
            num_samples = len(dataset)
            return range(num_samples)
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, n_obs=None, indices=None):
        """
        Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        if indices is None:
            indices = self.shuffled_indices(dataset)
        indices = indices[:n_obs]
        return dataset.select(indices)

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, self.config, split=split)

    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def map_dataset(self, dataset, add_prefix):
        mylogs.bp("map")
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                           remove_columns=dataset.column_names,
                           load_from_cache_file=False)

    def get(self, split, add_prefix=True, n_obs=None, split_validation_test=False, lang=None, file_name=None):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        mylogs.bp("get")
        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            if lang is not None:
                dataset = self.load_dataset(split=mapped_split, lang_code=lang)
            if file_name is not None:
                dataset = datasets.load_dataset(
                    'csv', data_files={split:file_name})[split]
            else:
                dataset = self.load_dataset(split=mapped_split)
            indices = self.get_split_indices(
                split, dataset, validation_size=len(dataset)//2)
            dataset = self.subsample(dataset, n_obs, indices)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            if lang is not None:
                dataset = self.load_dataset(split="train", lang_code=lang)
            if file_name is not None:
                dataset = datasets.load_dataset(
                    'csv', data_files={split:file_name})[split]
            else:
                dataset = self.load_dataset(split="train")
            indices = self.get_split_indices(
                split, dataset, validation_size=1000)
            dataset = self.subsample(dataset, n_obs, indices)
        else:
            mapped_split = self.split_to_data_split[split]
            if lang is not None:
                dataset = self.load_dataset(split=mapped_split, lang_code=lang)

            if file_name is not None:
                dataset = datasets.load_dataset(
                    'csv', data_files={split:file_name})[split]
            else:
                dataset = self.load_dataset(split=mapped_split)
            # shuffles the data and samples it.
            if n_obs is not None:
                dataset = self.subsample(dataset, n_obs)
        return self.map_dataset(dataset, add_prefix)

    ######### my template functions
    def fill_prompt(self, template, name, place_holder, plen = 0, num_holder="_i"):
        if plen==0: 
            plen = self.prompt_length 
            if name == "com":
                plen = self.common_length

        _pholder = place_holder
        place_holder = place_holder.replace("task", self.name)  
        place_holder = place_holder.replace("[", "<")  
        place_holder = place_holder.replace("]", ">")  
        while _pholder in template:
            if num_holder in _pholder:
                prompt = ""
                for i in range(plen):
                    token = place_holder
                    if num_holder != "_1":
                        token = token.replace(num_holder, "_" + str(i))  
                    else:
                        token = token.replace(num_holder, "")  
                    prompt += " " + token
            else:
                prompt = place_holder
            prompt = prompt.strip()
            for token in prompt.split():
                if not name in self.prompt_set:
                    self.prompt_set[name] = []
                if not token in self.prompt_set[name]:
                    self.prompt_set[name].append(token)
            template = template.replace(_pholder,prompt, 1)
        return template

    def fill_prompt_regex(self, template, regex):
        m = re.search(regex, template)
        while m: 
            if len(m.groups()) == 2:
                name = m.groups()[0]
                emb = m.groups()[1]
                plen = "1"
                if emb.isdigit():
                    plen = emb
                num_holder = "_" + str(plen)
                if emb == "i":
                    plen = 0
                    num_holder = "_i"
                place_holder = "[" + name + "_" + emb + "]"
                if plen != 0:
                    plen = [int(plen)]
                if name == "task":
                    name = self.name
                template = self.fill_prompt(template, name, place_holder, plen=plen, 
                        num_holder=num_holder)
                m = re.search(regex, template)
        return template

    def fill_prompts(self, template):
        mylogs.bp("fill_prompt")
        template = self.fill_prompt_regex(template, "\[([@a-zA-Z]+)_(\d+)\]")
        template = self.fill_prompt_regex(template, "\[([@a-zA-Z]+)_([a-zA-Z\?\d]+)\]")
        return template

    def get_prompts(self):
        data = {"task": self.name}
        self.fill_template(data)
        return self.prompt_set

    def get_template(self):
        tn = self.template
        src = "{task}: {source} (com) (prompt) (nat) (mask)" 
        target = "(mask) {target}"
        if "pre-" in tn:
            src = "(prompt) {source} (mask)" 
        if "unsup" in tn:
           src = src.replace("(mask)", "{mask}")
           target = target.replace("(mask)","{mask}")
        elif "sup" in tn:
           src = src.replace("(mask)", "")
           target = target.replace("(mask)","")
        if "-com" in tn:
           src = src.replace("(com)", "[com_i]")
        if "-pt-t" in tn:
           src = src.replace("(prompt)", "[task_i]")
        if "-pt-w" in tn:
           src = src.replace("(prompt)", "{prompt_fw}")
        if "-pt-c" in tn:
           src = src.replace("(prompt)", "{prompt_fwc}")
        if "-pt-sh" in tn:
           src = src.replace("(prompt)", "{prompt_sh}")
        if "-nat" in tn: 
           src = src.replace("(nat)", ", {rel_nat}")

        return src, target

    def extend_data(self, data):
        mylogs.bp("data")
        if "task" in data:
            task = data["task"]
            data["rel_tok"] = REL_TO_TOKEN[task] if task in REL_TO_TOKEN else task
            data["rel_word"] = REL_TO_WORD[task] if task in REL_TO_WORD else task
            data["rel_nat"] = REL_TO_PHRASE[task] if task in REL_TO_PHRASE else task
            rel_fw = REL_TO_PHRASE[task] if task in REL_TO_PHRASE else task
            rel_fw = rel_fw.split()
            prompts_fw = ["[task_" + w + "]" for w in rel_fw]
            data["prompt_fw"] = " ".join(prompts_fw)
            rel_sh = REL_TO_SHARED_TOKENS[task] if task in REL_TO_SHARED_TOKENS else task
            rel_sh = rel_sh.split()
            prompts_sh = ["[" + w + "_" + w + "]" for w in rel_sh]
            data["prompt_sh"] = " ".join(prompts_sh)
            #rel_fw_cycle = list(islice(cycle(rel_fw), self.prompt_length))
            prompts_fw_cycle = []
            for i in range(self.prompt_length):
                j = i % len(rel_fw)
                tok = "[task" + "_" + rel_fw[j] + "?" + str(i) + "]"
                prompts_fw_cycle.append(tok)
            data["prompt_fwc"] = " ".join(prompts_fw_cycle)
        return data

    def fill_template(self, data):
        mylogs.bp("fill")
        src,tgt = self.get_template()
        # remove unused place holders
        src = re.sub(r'\(.*?\)','',src)
        src = re.sub(' +', ' ',src)
        tgt = re.sub(r'\(.*?\)','',tgt)

        mask = "<extra_id_0>"
        data = self.extend_data(data)
        data["mask"] = mask
        data = defdict(data)
        # fill the templates with data
        src_texts = src.format_map(data)
        tgt_texts = tgt.format_map(data)
        src_texts = self.fill_prompts(src_texts)
        return src_texts, tgt_texts 

    def seq2seq_format(self, sources: List[str],
                       targets: List[str],
                       add_prefix: bool = False,
                       prefix: str = None,
                       extra_fields={}):
        src_prefix = self.name if prefix is None else prefix
        mylogs.bp("format")
        add_prefix = self.task_args.setdefault("add_prefix", False)
        orig_src = ' '.join(sources)
        sources = [src_prefix]+sources if add_prefix else sources
        src = ' '.join(sources)
        tgt =  ' '.join(targets)
        data = {'source': src,
                'target': tgt,
                'task': self.name,
                ** extra_fields}
        extra_fields["event"] = orig_src 
        extra_fields["tail"] = tgt 
        extra_fields["sel"] = False
        src_text, tgt_text = self.fill_template(data) 
        extra_fields["query"] = src_text
        extra_fields["resp"] = tgt_text
        if not "examples" in self.counter:
            self.counter["examples"] = 1
        if self.counter["examples"] < 5:
            mylogs.vlog.info("==============================================")
            mylogs.vlog.info("%s", extra_fields)
            self.counter["examples"] += 1
        return {'source': src_text,
                'target': tgt_text, 
                'task': self.name,
                'extra_fields': extra_fields}

class Squad(AbstractTask):
    name = "squad"
    metric = [metrics.squad]

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, split=split)

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answers']).split("\t")
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, add_prefix)


class DROP(AbstractTask):
    name = "drop"
    metric = [metrics.squad]

    def load_dataset(self, split):
        return datasets.load_dataset("drop", split=split)

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answers_spans']['spans'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['passage'])
        source = ["question:", question,
                  "context:", context]
        target = [answer]
        return self.seq2seq_format(source, target, add_prefix)


class PIQA(AbstractTask):
    name = "piqa"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('piqa', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example['goal'], "choice1:",
                     example["sol1"][0], "choice2:", example["sol2"][0]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class CommonsenseQA(AbstractTask):
    name = "commonsense_qa"
    labels_list = ["0", "1", "2", "3", "4"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('commonsense_qa', split=split)

    def preprocessor(self, example, add_prefix=True):
        label2id = {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4"}
        src_texts = ["question:", example['question'], "choice1:", example["choices"]["text"][0], "choice2:", example["choices"]["text"][1],
                     "choice3:", example["choices"]["text"][2], "choice4:", example["choices"]["text"][3], "choice5:", example["choices"]["text"][4]]
        tgt_texts = [label2id[example["answerKey"]]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SocialIQA(AbstractTask):
    name = "social_i_qa"
    labels_list = ["0", "1", "2"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('social_i_qa', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example['question'], "context:", example["context"], "|| choice0:",
                     example["answerA"][0], "|| choice1:", example["answerB"][0], "|| choice2:", example["answerC"][0]]
        tgt_texts = [str(int(example["label"]) - 1)]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SciTail(AbstractTask):
    name = "scitail"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('scitail', "snli_format", split=split)

    def preprocessor(self, example, add_prefix=True):
        label2id = {"entailment": "0", "neutral": "1"}
        src_texts = ["premise:", example['sentence1'],
                     "hypothesis:", example["sentence2"]]
        tgt_texts = [label2id[example["gold_label"]]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MRPC(AbstractTask):
    name = "mrpc"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score_with_invalid, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mrpc', split=split) 

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class COLA(AbstractTask):
    name = "cola"
    labels_list = ["0", "1"]
    metric = [metrics.matthews_corrcoef]
    metric_names = ["matthews_correlation"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'cola',
                                     split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

class SST2(AbstractTask):
    name = "sst2"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'sst2',
                                     split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class YelpPolarity(AbstractTask):
    name = "yelp_polarity"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "test": "test"}

    def load_dataset(self, split):
        print(split)
        return datasets.load_dataset('yelp_polarity')[split]

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class Amazon_Polarity(AbstractTask):
    name = "amazon_polarity"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('yelp_polarity', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", "<title> {0} <context> {1}".format(
            example['title'], example['context'])]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class STSB(AbstractTask):
    name = "stsb"
    labels_list = [str(np.round(label, decimals=1))
                   for label in np.arange(0, 5.2, 0.2)]
    metric = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    metric_names = ["pearson", "spearmanr"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb',
                                     split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

import os.path as op

import pandas as pd
class Atomic(AbstractTask):
    name = "atomic"
    metric = [metrics.rouge]
    metric_names = ["rouge"]
    do_shuffle = True
    samples_per_head = 3
    rels = []
    def __init__(self, config, task_args):
        super().__init__(config, task_args)
        self.data_path = task_args.data_path
        if not self.rels:
            self.rels = [self.name]

    def get_data_path(self, split):
        path = self.data_path
        if not path.startswith("/"):
            path= op.join(mylogs.home, self.data_path)
        if split == "test":
            mylogs.bp("path")
            if self.config == "full-test":
                path = op.join(path, self.config, self.name  + '.tsv')
            else:
                path = op.join(path, self.config, split  + '.tsv')
            print("TEST PATH:", path)
        else:
            path = op.join(path, split + '.tsv')
        return path

    def load_dataset(self, split):
        if split != "train":
            self.do_shuffle = False
        path = self.get_data_path(split)
        df = pd.read_table(path)
        df = self.filter(df, split)
        df = self.preproc_df(df, split)
        assert len(df) > 0, "data frame is empty for " + split + " of " + self.name + " " + path
        ds = Dataset.from_pandas(df)
        self.df = df
        return ds

    def check_n_obs(self, n_obs, total_size):
        if n_obs < 0:
            return total_size
        df = self.df
        lst = df['input_text'].value_counts()[:n_obs].index
        out = df[df['input_text'].isin(lst)]
        #m = pd.Series(range(0, n_obs), index=lst)
        #out = df[df['input_text'].isin(lst)].sort_values('input_text', key=lambda x: m[x])
        n_obs = len(out)
        return n_obs

    def preproc_df(self, df, split):
        df["freqs"] = df.groupby(['prefix','input_text'])['input_text'].transform('count')
        print("len df:", len(df))
        df = df.groupby(["prefix", "input_text"]).head(self.samples_per_head)
        print("len new df:", len(df))
        sort_by = ["freqs","input_text", "prefix"] 
        if "sel" in df:
            mylogs.bp("df")
            sort_by = ["sel", "freqs","input_text", "prefix"] 
        df = df.sort_values(by=sort_by, ascending=False)
        return df

    def filter(self, df, split):
        cond = ""
        for val in self.rels: 
            cond += f"| (df['prefix'] == '{val}') "
        cond = cond.strip("|")
        if cond: df = df[eval(cond)]
        return df


    #### ppppppppppppppp 
    def preprocessor(self, example, add_prefix=True):
        mylogs.bp("task_prep")
        src_texts = [str(example["input_text"])]
        tgt_texts = [str(example["target_text"])]
        extra_fields = {}
        extra_fields["event"] = example["input_text"]
        extra_fields["tail"] = example["target_text"]
        extra_fields["sel"] = example["sel"] if "sel" in example else False
        return self.seq2seq_format(src_texts, tgt_texts, 
                add_prefix=False, extra_fields=extra_fields)

class xIntent(Atomic):
    name = "xIntent"

class AtomicRel(Atomic):
    name = "atomic-rels"
    samples_per_rel = 100
    def __init__(self, config, task_args):
        super().__init__(config, task_args)
        self.train_samples_per_rel = task_args.train_samples
        self.val_samples_per_rel = task_args.val_samples
        self.test_samples_per_rel = task_args.test_samples
        self.rels = task_args.rels

    def get_data_path(self, split):
        path = self.data_path
        if not path.startswith("/"):
            path= op.join(mylogs.home, self.data_path)
        path = op.join(path, split + '.tsv')
        return path

    def preproc_df(self, df, split):
        if split == "train":
            samples_per_rel = self.train_samples_per_rel
        elif split == "validation":
            samples_per_rel = self.val_samples_per_rel
        else:
            samples_per_rel = self.test_samples_per_rel
        print("len df:", len(df))
        df = df.groupby(["prefix"]).head(samples_per_rel)
        print("len new df:", len(df))
        return df

    def check_n_obs(self, n_obs, total_size):
        return total_size

    def get_template(self):
        tn = self.template
        src = "{input_text} |(prompt)| {target_text}" 
        if "unsup" in tn:
           src = "{input_text} (prompt) {mask} {target_text}" 
        if "-pt-t" in tn:
           src = src.replace("(prompt)", "[task_i]")
        if "-pt-w" in tn:
           src = src.replace("(prompt)", "{prompt_fw}")
        if "-rel" in tn:
           target = "{prefix}"
        elif "-tok" in tn:
           target = "{rel_tok}"
        elif "-nat" in tn:
           target = "{rel_nat}"
        elif "-word" in tn:
           target = "{rel_word}"
        else:
           raise ValueError("Invalid template " + tn)
        if "unsup" in tn:
            target = "{mask}" + target
        return src, target

class xAttr(Atomic):
    name = "xAttr"

class xNeed(Atomic):
    name = "xNeed"

class xReact(Atomic):
    name = "xReact"

class oReact(Atomic):
    name = "oReact"

class xWant(Atomic):
    name = "xWant"

class oWant(Atomic):
    name = "oWant"

class xEffect(Atomic):
    name = "xEffect"

class oEffect(Atomic):
    name = "oEffect"

class QQP(AbstractTask):
    name = "qqp"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score_with_invalid, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qqp',
                                     split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question1:", example['question1'],
                     "question2:", example["question2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mnli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SNLI(AbstractTask):
    name = "snli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('snli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis: ", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MultiNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('multi_nli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QNLI(AbstractTask):
    name = "qnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qnli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example['question'],
                     "sentence:", example["sentence"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class RTE(AbstractTask):
    name = "rte"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'rte',
                                     split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class WNLI(AbstractTask):
    name = "wnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'wnli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUEBoolQ(AbstractTask):
    name = "superglue-boolq"
    labels_list = ['0', '1']
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'boolq', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"],
                     "passage:", example["passage"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUERTE(AbstractTask):
    name = "superglue-rte"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'rte', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUECB(AbstractTask):
    name = "superglue-cb"
    labels_list = ['0', '1', '2']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.mean_multiclass_f1(num_classes=3), metrics.accuracy]
    metric_names = ["f1_multiclass", "accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'cb', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUECOPA(AbstractTask):
    name = "superglue-copa"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'copa', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"],
                     "choice1:", example["choice1"],
                     "choice2:", example["choice2"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUEMultiRC(AbstractTask):
    name = "superglue-multirc"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.multirc_f1_over_all_answers,
              metrics.mean_group_metric(metrics.exact_match)]
    metric_names = ["f1", "em"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'multirc', split=split)

    def remove_markup(self, text):
        """Removes the HTML markup."""
        text = re.sub('<br>', ' ', text)
        text = re.sub('<(/)?b>', '', text)
        return text

    def preprocessor(self, example, add_prefix=True):
        group = example['idx']['question']
        # T5 applies remove_markup to the joined string, but this should not make
        # any difference as well.
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/a1352e625db7ec114062f99d99b0565b9e45c155/t5/data/preprocessors.py#L797
        src_texts = ["question:", self.remove_markup(example["question"]),
                     "answer:", self.remove_markup(example["answer"]),
                     "paragraph:", self.remove_markup(example["paragraph"])]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, extra_fields={"group": group})


class SuperGLUEWIC(AbstractTask):
    name = "superglue-wic"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'wic', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example["sentence1"],
                     "sentence2:", example["sentence2"],
                     "word:", example["word"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUEWSCFixed(AbstractTask):
    # source: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py
    """Convert WSC examples to text2text format.
     WSC includes a sentence along with 2 'spans': the first denoting a noun and
     the other a pronoun. The 'label' specifies whether or not the pronoun is
     referencing the noun. This preprocessor puts ' * ' around the noun and ' # '
     around the pronoun.
     For example, a typical example from WSC might look like
     {
         'text': 'This is a test sentence .',
         'span1_text': 'test',
         'span1_index': 3,
         'span2_text': 'This',
         'span2_index': 0,
         'label': 0
     }
     This example would be transformed to
     {
         'inputs': 'wsc text: # This # is a * test * sentence .',
         'targets': 'False'
     }
    """
    name = "superglue-wsc.fixed"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'wsc.fixed', split=split)

    def _mark_span(self, text, span_str, span_idx, mark):
        pattern_tmpl = r'^((?:\S+\s){N})(W)'
        pattern = re.sub('N', str(span_idx), pattern_tmpl)
        pattern = re.sub('W', span_str, pattern)
        return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)

    def preprocessor(self, example, add_prefix=True):
        # converts text as done in T5.
        text = example['text']
        text = self._mark_span(
            text, example['span1_text'], example['span1_index'], '*')
        # Compensate for 2 added "words" added in previous step.
        span2_index = example['span2_index'] + 2 * \
            int(example['span1_index'] < example['span2_index'])
        text = self._mark_span(text, example['span2_text'], span2_index, '#')
        src_texts = ["text:", text]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUERecord(AbstractTask):
    """Convert ReCoRD examples to text2text examples.
    ReCoRD contains a passage, query containing a '@placeholder' string, and a set
    of entities that are the possible values of the placeholder. Each train and
    validation example will have a list of answers, any of which would be
    considered correct.
    For example, a typical example from ReCoRD might look like
    {
      'passsage': 'This is the passage.',
      'query': 'A @placeholder is a bird.',
      'entities': ['penguin', 'potato', 'pigeon'],
      'answers': ['penguin', 'pigeon'],
    }
    which this preprocessor would turn into the following two examples:
    {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'penguin',
    }
    and
    {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'pigeon',
    }
    """
    name = "superglue-record"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.squad]
    metric_names = ["squad"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'record', split=split)

    def preprocessor(self, batch, add_prefix=True):
        new_batch = collections.defaultdict(list)
        keys = batch.keys()
        for values in zip(*batch.values()):
            ex = {k: v for k, v in zip(keys, values)}
            # updates the passage.
            passage = ex['passage']
            passage = re.sub(
                r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
            passage = re.sub(r'\n@highlight\n', '. ', passage)
            inputs = f"record query: {ex['query']} entities: {', '.join(ex['entities'])} passage: {passage}"
            if add_prefix:
                inputs = self.name + " " + inputs
            # duplicates the samples based on  number of answers.
            num_answers = len(ex["answers"])
            num_duplicates = np.maximum(1, num_answers)
            new_batch["source"].extend([inputs] * num_duplicates)
            new_batch["target"].extend(
                ex["answers"] if num_answers > 0 else ["<unk>"])
            new_batch["task"].extend([self.name] * num_duplicates)
            new_batch["extra_fields"].extend(
                [{"answers": ex["answers"]}]*num_duplicates)
        return new_batch

    def map_dataset(self, dataset, add_prefix=True):
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                           batched=True, remove_columns=dataset.column_names)


class WinoGrande(AbstractTask):
    name = "winogrande"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('winogrande', "winogrande_xl", split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example["sentence"],
                     "option0:", example["option1"],
                     "option1:", example["option1"]]
        tgt_texts = [str(int(example["answer"]) - 1)]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class PAWS(AbstractTask):
    name = "paws"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('paws', 'labeled_final', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


TASK_MAPPING = OrderedDict(
    [
        ('atomic', Atomic),
        ('xIntent', xIntent),
        ('xAttr', xAttr),
        ('xNeed', xNeed),
        ('xReact', xReact),
        ('oReact', oReact),
        ('xWant', xWant),
        ('oWant', oWant),
        ('xEffect', xEffect),
        ('oEffect', oEffect),
        ('atomic-rels', AtomicRel),
        ('squad', Squad),
        ('mrpc', MRPC),
        ('cola', COLA),
        ('sst2', SST2),
        ('qnli', QNLI),
        ('rte', RTE),
        ('wnli', WNLI),
        ('mnli', MNLI),
        ('qqp', QQP),
        ('stsb', STSB),
        ('superglue-boolq', SuperGLUEBoolQ),
        ('superglue-rte', SuperGLUERTE),
        ('superglue-cb', SuperGLUECB),
        ('superglue-copa', SuperGLUECOPA),
        ('superglue-multirc', SuperGLUEMultiRC),
        ('superglue-wic', SuperGLUEWIC),
        ('superglue-wsc.fixed', SuperGLUEWSCFixed),
        ('superglue-record', SuperGLUERecord),
        ('multi_nli', MultiNLI),
        ('snli', SNLI),
        ('piqa', PIQA),
        ('drop', DROP),
        ('newsqa', Squad),
        ('searchqa', Squad),
        ('triviaqa', Squad),
        ('nq', Squad),
        ('hotpotqa', Squad),
        ("social_i_qa", SocialIQA),
        ("commonsense_qa", CommonsenseQA),
        ("winogrande", WinoGrande),
        ("scitail", SciTail),
        ('yelp_polarity', YelpPolarity),
        ('amazon_polarity', Amazon_Polarity),
        ('paws', PAWS),
    ]
)


class AutoTask:
    @classmethod
    def get(self, task, config, task_args=None):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, task_args)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
