import wandb
import seaborn as sns
#import PIL
import matplotlib.pyplot as plt
from transformers.integrations import WandbCallback
from math import floor
import attempt.mylogs as mylogs
from attempt.myutil import tag_to_image
import matplotlib.pyplot as plt
import json

class WBCallback(WandbCallback):
    cur_epoch = -1
    def __init__(self, **kwargs):
        super().__init__()

    def setup(self, args, state, model, **kwargs):
        epoch = floor(state.epoch)
        mylogs.bp("wand")
        if epoch != self.cur_epoch: # and state.global_step > 0:
            self.cur_epoch = epoch
            encoder = model.encoder
            np_scores = encoder.attn_scores.detach().cpu().numpy()
            #fig = plt.imshow(np_scores, cmap='hot', interpolation='nearest')
            labels = model.encoder.prompt_names
            fig, axes = plt.subplot_mosaic("ABB")
            ax1, ax2 = axes["A"], axes["B"]
            ax1.set_title(f"Epoch:{epoch}-{state.epoch}  Step:{state.global_step}")
            fig.set_size_inches(12.5, 6.5)
            ax1.axis("off")
            img = tag_to_image()
            # ax2.axis("off")
            fig.figimage(img, 5, 100)
            #fig.figimage(self.tag_img, 5, 120)
            sns.heatmap(np_scores, ax=ax2, cmap="crest", annot=True, 
                    xticklabels=labels,
                    yticklabels=labels,
                    linewidth=0.5)
            #plt.tight_layout()
            mylogs.bp("wand")
            wandb.log({"attn_scores":wandb.Image(fig)})
            if encoder.attn_method == "rb":
                ax2.clear()
                np_router = encoder.router.detach().cpu().numpy()
                labels = model.encoder.prompt_names
                sns.heatmap(np_router, cmap="crest", annot=False, ax=ax2, 
                        xticklabels=labels,
                        yticklabels=labels,
                        linewidth=0.5)
                wandb.log({"router":wandb.Image(fig)})
            mylogs.bp("wand")
            plt.close("all")

