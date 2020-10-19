# %%
import os
import re
import pandas as pd
from pathlib import Path
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from skimage import io
import matplotlib as mpl 
import matplotlib.pyplot as plt

# %%
class OdDataset:

    def __init__(self, img_size=512, obj_size=28, max_obj_per_img=5, obj_type="mnist"):
        self.img_size = img_size
        self.obj_size = obj_size
        self.max_obj = max_obj_per_img

        if obj_type == "mnist":
            dataset = torchvision.datasets.MNIST(
                root="./data", download=True, train=True
                )
            self.data = dataset.data
            self.labels = dataset.train_labels
        else:
            raise ValueError("Dataset type not implemented")

    def generate_dataset(self, out_dir="./dataset", sets="train-val", n=100, prop_train=0.8):
        n_train = int(n * prop_train)
        out_dir = Path(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir/"train", exist_ok=True)
        os.makedirs(out_dir/"valid", exist_ok=True)

        for i in range(n_train):
            coords, img = self.generate_img()
            self.save_img_coord(out_dir/"train", f"img_{i}", coords, img)
        
        for i in range(n - n_train):
            coords, img = self.generate_img()
            self.save_img_coord(out_dir/"valid", f"img_{i}", coords, img)

    def generate_img(self):
        img = np.zeros((self.img_size, self.img_size))
        n_trgt = np.random.randint(1, self.max_obj) if self.max_obj > 1 else self.max_obj
        trgt_id = np.random.choice(len(self.data), size=n_trgt, replace=False)
        coords = []
        for idx in trgt_id:
            xr, yr = np.random.randint(0,self.img_size-self.obj_size, 2)
            cls_id = self.labels[idx].item()
            coord = [xr, yr, xr + self.obj_size, yr + self.obj_size]
            coords.append([cls_id, *coord])
            trgt = transform.resize(self.data[idx], (self.obj_size, self.obj_size))
            img[coord[1]: coord[3], coord[0]: coord[2]] = trgt

        return np.array(coords), img
    
    @classmethod
    def save_img_coord(self, dest, name, coords, img):
        io.imsave(dest / f"{name}.png", (img*255).astype(np.uint8))
        with open(dest / f"{name}.txt", "w+") as f:
            for coord in coords:
                f.write(",".join([str(x) for x in coord]))
                f.write("\n")


def plot_targets(name):
    img = io.imread(name)
    with open(re.sub(".png", ".txt", name), "r") as f:
        annots = [ [int(y) for y in x.split(",")] for x in f.readlines()]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img)
    for annot in annots:
        plt.text(annot[1]+3, annot[2]-6, str(annot[0]), backgroundcolor="green", fontsize=6)
        w, h = (-annot[1] + annot[3]), (-annot[2] + annot[4])
        patch = mpl.patches.Rectangle((annot[1], annot[2]), w, h, facecolor="none", linewidth=2, edgecolor="green")
        ax.add_patch(patch)
    plt.show()

def plot_box(img, coords, names=None, mod="xyxy"):
    fig, ax = plt.subplots()
    ax.imshow(img)
    if mod == "xyxy":
        for i, annot in enumerate(coords):
            if names:
                ax.text(annot[0]+3, annot[1]-6, str(names[i]), backgroundcolor="green", fontsize=6)
            w, h = (-annot[0] + annot[2]), (-annot[1] + annot[3])
            patch = mpl.patches.Rectangle((annot[0], annot[1]), w, h, facecolor="none", linewidth=2, edgecolor="green")
            ax.add_patch(patch)
    elif mod == "xywh":
        for i, annot in enumerate(coords):
            x, y = round(annot[0] - annot[2]/2), round(annot[1] - annot[3]/2)
            if names:
                ax.text(x+3, y-6, str(names[i]), backgroundcolor="green", fontsize=6)
            patch = mpl.patches.Rectangle((x, y), annot[2], annot[3], facecolor="none", linewidth=2, edgecolor="green")
            ax.add_patch(patch)




# %%
if __name__ == "__main__":
    generator = OdDataset(256, 56, 1)
    generator.generate_dataset(n=1500)
    plot_targets("/Users/brown08/Documents/code_proj/yolomy/dataset/train/img_70.png")

# %%
