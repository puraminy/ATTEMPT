from comet.train.common import *
import pandas as pd
from datetime import datetime
from pathlib import Path
import random
import comet.train.mylogs as logs

class Template:
    def __init__(self, rel, temp_num=1):
        self.rel = rel
        self.prompts = []
           
    def fill_consts(self, template):
        text = self.fill_prompt_regex(text, rel, "{([@a-zA-Z]+)_(\d+)}")
        text = self.fill_prompt_regex(text, rel, "{([@a-zA-Z]+)_([a-zA-Z]+)}")

    def fill_prompt_regex(self, text, row_rel, regex):
        m = re.search(regex, text)
        while m: 
            if len(m.groups()) == 2:
                rel = m.groups()[0]
                emb = m.groups()[1]
                plen = "1"
                if emb.isdigit():
                    plen = emb
                num_holder = "_" + str(plen)
                if emb == "i":
                    plen = 0
                    num_holder = "_" + emb
                place_holder = "{" + rel + "_" + emb + "}"
            elif len(m.groups()) == 3:
                rel = m.groups()[0]
                emb = m.groups()[1]
                plen = m.groups()[2]
                num_holder = "_" + plen
                place_holder = "{" + rel + "_" + emb + "_" + plen + "}"
            if plen != 0:
                plen = [int(plen)]
            if rel == "rel":
                rel = row_rel
            text = self.fill_prompt(text, rel, place_holder, plen=plen, 
                    num_holder=num_holder, row_rel=row_rel)
            m = re.search(regex, text)
        return text

    def fill_prompt(self, text, rel, place_holder, counter = 0, lang="", plen = 0, num_holder="_i", row_rel=""):
        if not row_rel: row_rel = rel
        pi = 0
        if plen==0: 
            if rel in relation_prompt_lengths:
                plen = relation_prompt_lengths[rel]
            else:
                if False: #"merge" in rel or "mat" in rel:
                    plen = [self.num_prompts]  
                else:
                    plen = [self.num_prompt_tokens]  
        _pholder = place_holder
        place_holder = place_holder.replace("{", "<")  
        place_holder = place_holder.replace("}", ">")  
        place_holder = place_holder.replace("rel", row_rel)  
        place_holder = place_holder.replace("lang", lang)  
        #dlog.info("text: %s", text)
        while _pholder in text:
            if num_holder in _pholder:
                enc_plen = plen[pi] if pi < len(plen) else plen[-1] 
                prompt = ""
                for i in range(counter, counter + enc_plen):
                    token = place_holder
                    if num_holder != "_1":
                        token = token.replace(num_holder, "_" + str(i))  
                    else:
                        token = token.replace(num_holder, "")  
                    prompt += " " + token
            elif _pholder == "{tokens}": 
                prompt = rel_nat_maps[rel]["tokens"]
            elif _pholder == "{tokens-rand}": 
                permute = rel_nat_maps[rel]["tokens"].split()
                random.shuffle(permute)
                prompt = " ".join(permute)
            else:
                #mlog.info("************** using tokens of pholder %s",_pholder)
                prompt = place_holder
            prompt = prompt.strip()
            enc_plen = len(prompt.split())
            for token in prompt.split():
                if not rel in self.encoder_prompts:
                    self.encoder_prompts[rel] = []
                if not token in self.encoder_prompts[rel]:
                    self.encoder_prompts[rel].append(token)
            text = text.replace(_pholder,prompt, 1)
            counter += enc_plen 
            pi += 1
        #dlog.info("text: %s", text)
        return text

