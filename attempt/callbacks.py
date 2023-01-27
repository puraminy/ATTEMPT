import wandb
import seaborn as sns
#import PIL
import matplotlib.pyplot as plt
from transformers.integrations import WandbCallback
from math import floor

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
            plt.tight_layout()
            ax.set_title("attend_source:" + str(encoder.attend_source) \
                      + " | attend_input:" + str(encoder.attend_input) \
                      + " | attend_target:" + str(encoder.attend_target))
            wandb.log({"attn_scores":wandb.Image(ax)})
            plt.close("all")

