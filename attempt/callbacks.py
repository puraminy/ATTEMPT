import wandb
import seaborn as sns
#import PIL
import matplotlib.pyplot as plt
from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerCallback 
from math import floor
import attempt.mylogs as mylogs
from attempt.myutil import tag_to_image
import matplotlib.pyplot as plt
import json, os

class AnnealCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        mylogs.bp("anneal")
        model = kwargs.pop("model", None)
        e = model.encoder
        e.anneal(state.global_step)
        wandb.log({"router_temperature": e.router_temperature})
        mylogs.winfo("router","%s: %s  (%s %s > %s)", state.global_step, 
                e.router_temperature, e.anneal_dir, e.anneal_rate, enc.anneal_min)

class WBCallback(WandbCallback):
    cur_epoch = -1
    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def save_images(scores, labels, state=None, fname=""):
        np_scores = scores.detach().cpu().numpy()
        fig, axes = plt.subplot_mosaic("ABB")
        ax1, ax2 = axes["A"], axes["B"]
        if state is not None:
            ax1.set_title(f"Epoch:{state.epoch}  Step:{state.global_step} Best:{state.best_metric}")
        else:
            ax1.set_title(fname)
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
        wandb.log({fname:wandb.Image(fig)})
        plt.close("all")

    def setup(self, args, state, model, **kwargs):
        epoch = floor(state.epoch)
        mylogs.bp("wand")
        epoch = int(epoch)
        if epoch % 10 == 0 or state.global_step == 1:
            self.cur_epoch = epoch
            labels = model.encoder.prompt_names
            scores = model.encoder.attn_scores
            WBCallback.save_images(scores, labels, state, fname="attn_scores")
