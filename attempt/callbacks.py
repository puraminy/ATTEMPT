import wandb
import seaborn as sns
#import PIL
import matplotlib.pyplot as plt
from transformers.integrations import WandbCallback
from math import floor
import attempt.mylogs as mylogs

class WBCallback(WandbCallback):
    cur_epoch = -1
    def setup(self, args, state, model, **kwargs):
        epoch = floor(state.epoch)
        if epoch != self.cur_epoch and state.global_step > 0:
            self.cur_epoch = epoch
            encoder = model.encoder
            np_scores = encoder.attn_scores.detach().cpu().numpy()
            #fig = plt.imshow(np_scores, cmap='hot', interpolation='nearest')
            labels = model.encoder.prompt_names
            ax = sns.heatmap(np_scores, cmap="crest", annot=True, 
                    xticklabels=labels,
                    yticklabels=labels,
                    linewidth=0.5)
            title = mylogs.get_tag(as_str=True)
            if not "attend_source" in title: 
                title += " | attend_source:" + str(encoder.attend_source) 
            if not "attend_input" in title: 
                title += " | attend_input:" + str(encoder.attend_input) 
            if not "attend_target" in title: 
                title += " | attend_target:" + str(encoder.attend_target) 
            if not "attn_method" in title: 
                title += " | attn_method:" + str(encoder.attn_method)
            ax.set_title(title)
            plt.tight_layout()
            mylogs.bp("wand")
            lname = "attn_scores_"+title.replace("|","_").replace(" ","_").replace(":","_")
            wandb.log({lname:wandb.Image(ax)})
            np_router = encoder.router.detach().cpu().numpy()
            #fig = plt.imshow(np_scores, cmap='hot', interpolation='nearest')
            labels = model.encoder.prompt_names
            ax = sns.heatmap(np_router, cmap="crest", annot=True, 
                    xticklabels=labels,
                    yticklabels=labels,
                    linewidth=0.5)
            ax.set_title(title)
            plt.tight_layout()
            lname = "router_"+title.replace("|","_").replace(" ","_").replace(":","_")
            wandb.log({lname:wandb.Image(ax)})
            mylogs.bp("wand")
            plt.close("all")

