from pathlib import Path

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

def image_from_text(texts,size_fnts, path_font=Path(__file__).parent / "seguiemj" / "seguiemj.ttf"):
    x = (int(max([len(l)*s for l,s in zip(texts,size_fnts)])*1)//16)*16
    y =  (int(sum(size_fnts)*1.7)//16)*16
    im = Image.new("RGBA", (x,y), (255, 255, 255, 0))
    pos = 0
    for text,size_fnt in zip(texts,size_fnts):
        pos += size_fnt//2
        fnt = ImageFont.truetype(str(path_font), size_fnt)
        with Pilmoji(im) as pilmoj:
            pilmoj.text((size_fnt//2,pos), text, font=fnt, fill=(0, 0, 0), embedded_color=True)
        pos += len(text.splitlines())*size_fnt
    return im

