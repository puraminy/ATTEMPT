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
        wandb.log({"temperature": e.temperature})
        #mylogs.winfo("router","%s: %s  (%s %s > %s)", state.global_step, 
        #        e.router_temperature, e.anneal_dir, e.anneal_rate, e.anneal_min)

class WBCallback(WandbCallback):
    cur_epoch = -1
    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def save_images(scores, x_labels, y_labels, state=None, fname="", annot=True):
        fig, axes = plt.subplot_mosaic("ABB;ACC;ADD")
        ax1, ax2, ax3,ax4 = axes["A"], axes["B"], axes["C"], axes["D"]
        axes = [ax2, ax3, ax4]
        if state is not None:
            ax2.set_title(f"Epoch:{state.epoch}  Step:{state.global_step} Best:{state.best_metric}")
        else:
            ax2.set_title(fname)
        fig.set_size_inches(12.5, 6.5)
        ax1.axis("off")
        img = tag_to_image()
        # ax2.axis("off")
        fig.figimage(img, 5, 100)
        #fig.figimage(self.tag_img, 5, 120)
        for score, ax in zip(scores, axes):
            np_score = score.detach().cpu().numpy()
            sns.heatmap(np_score, ax=ax, cmap="crest", annot=annot, 
                    annot_kws={'rotation': 90}, 
                    xticklabels=x_labels,
                    yticklabels=y_labels,
                    linewidth=0.5)
        #plt.tight_layout()
        mylogs.bp("wand")
        wandb.log({fname:wandb.Image(fig)})
        plt.close("all")

    def setup(self, args, state, model, **kwargs):
        epoch = floor(state.epoch)
        mylogs.bp("wand")
        epoch = int(epoch)
        if epoch % 10 == 1 or state.global_step == 2:
            self.cur_epoch = epoch
            p = "start" if state.global_step == 1 else "ep"
            x_labels = y_labels = model.encoder.prompt_names
            scores = model.encoder.attn_scores
            model.encoder.first_image = True
            #WBCallback.save_images(scores, x_labels, y_labels, 
            #                       state, fname= p + "_attn_scores")
