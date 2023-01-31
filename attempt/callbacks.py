import wandb
import seaborn as sns
#import PIL
import matplotlib.pyplot as plt
from transformers.integrations import WandbCallback
from math import floor
import attempt.mylogs as mylogs
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.transforms import IdentityTransform
import json
from io import BytesIO

def text_to_rgba(s, *, dpi, **kwargs):
    # To convert a text string to an image, we can:
    # - draw it on an empty and transparent figure;
    # - save the figure to a temporary buffer using ``bbox_inches="tight",
    #   pad_inches=0`` which will pick the correct area to save;
    # - load the buffer using ``plt.imread``.
    #
    # (If desired, one can also directly save the image to the filesystem.)
    fig = Figure(facecolor="white")
    fig.text(10, 0, s, **kwargs)
    with BytesIO() as buf:
        fig.savefig(buf, dpi=dpi, format="png", bbox_inches="tight",
                    pad_inches=0)
        buf.seek(0)
        rgba = plt.imread(buf)
    return rgba



class WBCallback(WandbCallback):
    cur_epoch = -1
    def __init__(self, **kwargs):
        super().__init__()
        tags = mylogs.get_full_tag()
        tag_labels = "=\n".join(list(tags.keys()))
        tag_values = "\n".join(list(tags.values()))
        self.tag_labels_img = text_to_rgba(tag_labels, color="blue", fontsize=14, dpi=100)
        self.tag_values_img = text_to_rgba(tag_values, color="green", fontsize=14, dpi=100)
        tag_dict_str = json.dumps(tags, indent=2)
        self.tag_dict_img = text_to_rgba(tag_dict_str, color="blue", fontsize=14, dpi=100)

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
            fig.set_size_inches(12.5, 6.5)
            ax1.axis("off")
            ax2.axis("off")
            fig.figimage(self.tag_dict_img, 5, 200)
            sns.heatmap(np_scores, ax=ax2, cmap="crest", annot=True, 
                    xticklabels=labels,
                    yticklabels=labels,
                    linewidth=0.5)
            #plt.tight_layout()
            mylogs.bp("wand")
            lname = "attn_scores"
            wandb.log({lname:wandb.Image(fig)})
            np_router = encoder.router.detach().cpu().numpy()
            labels = model.encoder.prompt_names
            ax2.clear()
            sns.heatmap(np_router, cmap="crest", annot=False, ax=ax2, 
                    xticklabels=labels,
                    yticklabels=labels,
                    linewidth=0.5)
            lname = "router"
            wandb.log({lname:wandb.Image(fig)})
            mylogs.bp("wand")
            plt.close("all")

