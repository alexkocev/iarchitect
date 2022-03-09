import base64
import operator
from pathlib import Path

import IPython
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pilmoji import Pilmoji


with open(Path(__file__).parent / "emoji.txt",encoding="utf8") as f:
    EMOJIS = f.read().splitlines()



def npemojis(number,with_empy = True):
    emojis = EMOJIS.copy()
    emojis = emojis * (number // len(emojis)) + emojis[:number % len(emojis)]

    while len(emojis)<number:
        emojis
    if with_empy:
        emojis = ['🟩'] + emojis

    return np.array(emojis)

def image_from_text(texts,size_fnts,
                    path_font=Path(__file__).parent / "seguiemj" / "seguiemj.ttf",
                    max_size=None):
    x = (int(max([len(l)*s for l,s in zip(texts,size_fnts)])*1)//16)*16
    y =  (int(sum(size_fnts)*1.7)//16)*16
    im = Image.new("RGBA", (x,y), (255, 255, 255, 255))
    pos = 0
    for text,size_fnt in zip(texts,size_fnts):
        pos += size_fnt//2
        fnt = ImageFont.truetype(str(path_font), size_fnt)
        with Pilmoji(im) as pilmoj:
            pilmoj.text((size_fnt//2,pos), text, font=fnt, fill=(0, 0, 0), embedded_color=True)
        pos += len(text.splitlines())*size_fnt
    # if max_size is not None:
    #     if max(im.size)>max_size:
    #         ratio = max_size/max(im.size)
    #         im = im.resize((int(ratio*im.size[0]),
    #                         int(ratio*im.size[1])
    #                         ))
    #         print(im.size)
    return im


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)


def create_list_gif(images,filename,fps):
    if Path(filename).suffix!=".gif":
        filename = filename + ".gif"
    imageio.mimwrite(filename, images, format= '.gif', fps = fps)

def create_list_video(images,filename,fps=30,embed=True):
    if Path(filename).suffix!=".mp4":
        filename = filename + ".mp4"
    sizes = np.array(list(map(operator.attrgetter("size"),images)))

    for xy in range(2):
        s = (slice(None,None,None),xy)
        print(sizes[s])
        if np.unique(sizes[s]).shape[0]>1:
            print(xy)
            print(sizes[s])
            print(np.unique(sizes[s],return_counts=True))
            images = list(
                map(operator.methodcaller("crop",
                                          (0,0)+tuple(sizes[np.argmin(sizes,axis=xy)[0]]))
                    ,images))
            sizes = np.array(list(map(operator.attrgetter("size"),images)))
    with imageio.get_writer(filename, fps=fps,mode="I") as video:
        for i,im in enumerate(images):
            print(f"{i} -> {filename}")
            print(np.array(im).shape)
            video.append_data(np.array(im))
    if embed:
        return embed_mp4(filename)
    return filename
