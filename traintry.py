# %%

import torch
import numpy as np
from models import ModelFromCFG
import os
from pathlib import Path
from skimage import io
import re
from dataset_generator import plot_targets, plot_box
from PIL import Image
from models import process_preds
from utils import bbox_iou, xywh2xyxy, xyxy2xywh
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

def collate_fn(batch):
    res = {
        "img":torch.tensor([x["img"] for x in batch]).unsqueeze(1),
        "annotations":[torch.tensor(x["annotations"]) for x in batch],
        "targets":[torch.tensor(x["targets"]) for x in batch]
    }
    return res

def comput_loss(proc_pred, annotations_gt, targets, iou_th=0.5, giou_ratio=0.5):
    #procpred = process_preds(model_out[0], int(np.sqrt(out.shape[1])) , 256, 56)
    boxloss, closs, objloss = torch.tensor([0]).float(), torch.tensor([0]).float(), torch.tensor([0]).float()
    for j in range(len(proc_pred)):
        for i, gt in enumerate(annotations_gt[j]):
            # get ious+
            ious = bbox_iou(gt.float(), xywh2xyxy(procpred[j,:,:4]).float())
            # get reelvant predictions
            pertinent = torch.where(ious>iou_th)[0]

            if len(pertinent):
                best_id = torch.max(ious[pertinent], 0)[1]
                best_bb = procpred[j, best_id, :]
                closs += pred_criterion(best_bb[5:].unsqueeze(0), torch.tensor(targets[i]))
                boxloss += (1 - ious[pertinent]).mean()
            
            trgt_objectness = (1 - giou_ratio) + giou_ratio * ious.detach().clamp(0)
            objloss += obj_criterion(procpred[j, ..., 4], trgt_objectness)
        
    loss = 2*boxloss + closs + 2*objloss
    loss_print = dict(box=boxloss.detach(), pred=closs.detach(), obj=objloss.detach())
    return loss, loss_print


class Dataset:

    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.filelist = [x for x in os.listdir(data_folder) if Path(x).suffix == ".png"]
        
    def __getitem__(self, idx):
        img = io.imread(self.data_folder / self.filelist[idx])/255
        with open(self.data_folder / re.sub(".png", ".txt", self.filelist[idx]), "r") as f:
            annots = [ [int(y) for y in x.split(",")] for x in f.readlines()]
        classes = [x.pop(0) for x in annots]
        return dict(img=img, annotations=annots, targets=classes)

    def __len__(self):
        return len(self.filelist)


# %%
epochs = 10
lr = 1e-4
iou_th = 0.5
giou_ratio = 0.2
anchor_size = 56

# %%
dataset_tr = Dataset("./dataset/train")
dataset_val = Dataset("./dataset/valid")
dataloader_train = DataLoader(dataset_tr, batch_size=3, shuffle=True, collate_fn=collate_fn, drop_last=True)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pred_criterion = CrossEntropyLoss()
obj_criterion = BCEWithLogitsLoss(reduction="mean")
model = ModelFromCFG("mockconfig.cfg")
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)


# %%
for epoch in range(epochs):
    losses = dict(box=[], obj=[], pred=[])
    for i, batch in enumerate(dataloader_train):

        imgs = batch["img"].float().to(device)
        annotations = [x.to(device) for x in batch["annotations"] ]
        targets = [x.to(device) for x in batch["targets"]]

        output = model(imgs)
        out_size = int(np.sqrt(output.shape[1]).item())
        procpred = process_preds(output, out_size , imgs.shape[-1], anchor_size)
        loss, loss_print = comput_loss(procpred, annotations, targets, iou_th, giou_ratio)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("".join(f"{k}: {v} " for k,v in loss_print.items()))
        for k, v in loss_print.items():
            losses[k].append(v)


fig, ax = plt.subplots(nrows=3, figsize=(5,6))
ax[0].plot(losses["obj"])
ax[1].plot(losses["box"])
ax[2].plot(losses["pred"])
plt.show()
