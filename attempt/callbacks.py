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
import io
from PIL import Image

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
    def save_images(scores, x_labels, y_labels, state=None, fname="", 
            annot=True,title="", add_tags=True):
        if not title: title = fname
        if add_tags:
            fig, axes = plt.subplot_mosaic("ABB;ACC;ADD")
            ax1, ax2, ax3,ax4 = axes["A"], axes["B"], axes["C"], axes["D"]
            axes = [ax2, ax3, ax4]
            ax_t = ax2
        else:
            fig, axes = plt.subplot_mosaic("A;B;C")
            ax1, ax2, ax3 = axes["A"], axes["B"], axes["C"]
            axes = [ax1, ax2, ax3]
            ax_t = ax1
        if state is not None:
            ax_t.set_title(f"Epoch:{state.epoch}  Step:{state.global_step} Best:{state.best_metric}")
        else:
            ax_t.set_title(title)
        fig.set_size_inches(12.5, 6.5)
        if add_tags:
            ax1.axis("off")
            tags = mylogs.get_full_tag()
            img = tag_to_image(tags)
            fig.figimage(img, 5, 100)
        for score, ax in zip(scores, axes):
            np_score = score.detach().cpu().numpy()
            sns.heatmap(np_score, ax=ax, cmap="crest", annot=annot, 
                    annot_kws={'rotation': 90}, 
                    xticklabels=x_labels,
                    yticklabels=y_labels,
                    linewidth=0.5)
        #plt.tight_layout()
        mylogs.bp("wand")
        if fname:
            wandb.log({fname:wandb.Image(fig)})
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close("all")
        return img_buf

    @staticmethod
    def save_image(score, x_labels, y_labels, fname="", 
            annot=True,title="", tags={}):
        if not title: title = fname
        if tags:
            fig, axes = plt.subplot_mosaic("AB")
            ax1, ax2 = axes["A"], axes["B"]
            ax_t = ax2
        else:
            fig, axes = plt.subplot_mosaic("A")
            ax1 = axes["A"]
            ax_t = ax1
        ax_t.set_title(title)
        fig.set_size_inches(12.5, 6.5)
        if tags:
            ax1.axis("off")
            img = tag_to_image(tags)
            fig.figimage(img, 5, 100)

        np_score = score.detach().cpu().numpy()
        sns.heatmap(np_score, ax=ax_t, cmap="crest", annot=annot, 
                annot_kws={'rotation': 90}, 
                xticklabels=x_labels,
                yticklabels=y_labels,
                linewidth=0.5)
        #plt.tight_layout()
        mylogs.bp("wand")
        if fname:
            wandb.log({fname:wandb.Image(fig)})
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close("all")
        return img_buf

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
