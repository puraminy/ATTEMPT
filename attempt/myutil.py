from io import BytesIO
from matplotlib.figure import Figure
from matplotlib.transforms import IdentityTransform
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as clr
import attempt.mylogs as mylogs
import json

def text_to_image(s, *, dpi, **kwargs):
    # To convert a text string to an image, we can:
    # - draw it on an empty and transparent figure;
    # - save the figure to a temporary buffer using ``bbox_inches="tight",
    #   pad_inches=0`` which will pick the correct area to save;
    # - load the buffer using ``plt.imread``.
    #
    # (If desired, one can also directly save the image to the filesystem.)
    fig = Figure(facecolor="none")
    fig.text(10, 0, s, **kwargs)
    with BytesIO() as buf:
        fig.savefig(buf, dpi=dpi, format="png", bbox_inches="tight",
                    pad_inches=0)
        buf.seek(0)
        rgba = plt.imread(buf)
    return rgba

def tag_to_image():
    tags = mylogs.get_full_tag()
    tags = dict(sorted(tags.items()))
    tag_labels = ":\n\n".join(list(tags.keys()))
    tag_values = "\n\n".join(list(tags.values()))
    tag_labels_img = text_to_image(tag_labels, color="gray", fontsize=14, dpi=100)
    tag_values_img = text_to_image(tag_values, color="blue", 
            weight="bold", fontsize=14, dpi=100)
    tag_dict_str = json.dumps(tags, indent=2)
    tag_dict_img = text_to_image(tag_dict_str, color="blue", fontsize=14, dpi=100)

    return tag_labels_img, tag_values_img, tag_dict_img

def df_to_image(df):
    # Set background to white
    tag_labels_img, tag_values_img, tag_dict_img = tag_to_image()
    fig, axes = plt.subplot_mosaic("ABB")
    ax1, ax2 = axes["A"], axes["B"]
    fig.set_size_inches(12.5, 6.5)
    ax1.axis("off")
    sns.heatmap(df, ax=ax2, annot=True, cbar=False)
    fig.figimage(tag_labels_img, 5, 120)
    fig.figimage(tag_values_img, 50, 100)
    return fig 
