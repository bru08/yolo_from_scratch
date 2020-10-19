# %%
import torch
from torch import nn
import re
import numpy as np
from torchsummary import summary

# %%

class YoloLayer(nn.Module):

    def __init__(self, batch_size, exp_filters):
        super(YoloLayer, self).__init__()
        self.bs = batch_size
        # self.nc = classes
        self.exp_filters = exp_filters

    def forward(self, x):
        res = x.view(self.bs, self.exp_filters, -1).permute(0, 2, 1)
        return res

    def __repr__(self):
        return(f"Yolo(batch_size={self.bs} exp_filters={self.exp_filters})")


def process_preds(pred_list, final_size, input_size, anchor_size):

    # cell size in original pixel units
    cell_size = input_size / final_size
    for preds in pred_list:
        # xy coordinate w.r.t. final feature map grid, in order to sum x,y for each cell location
        xyg = torch.tensor([divmod(i, final_size)[::-1] for i in range(preds.shape[0])])
        #reconstruct x,y for original
        preds[:,0:2] = cell_size * (preds[:,0:2].sigmoid() + xyg)
        # obtain width and height multiplying a reference size
        preds[:,2:4] = (anchor_size * preds[:, 2:4].exp()).clamp(max=1E3)
    return pred_list


class ModelFromCFG(nn.Module):

    def __init__(self, cfg_file, in_ch=1):
        super(ModelFromCFG, self).__init__()
        self.config = self.parse_config(cfg_file)
        self.modules_list = self.build_model(self.config)
        self.can = in_ch
    
    def forward(self, x):
        for i, mod_elem in enumerate(self.modules_list):
            x = mod_elem(x)
        return x
    
    def __repr__(self):
        summary(self, (self.can, 256, 256))
        return "Model structure and input shape (3, 256, 256)"

    def build_model(self, config):
        res = torch.nn.ModuleList()
        filters = 1
        for elem in config:
            if elem["title"] == "convolutional":
                res.append(
                    nn.Conv2d(filters,elem["filters"],3)
                )
                filters=elem["filters"]
            elif elem["title"] == "maxpool":
                res.append(
                    nn.MaxPool2d(2, stride=2)
                )
            elif elem["title"] == "linear":
                res.append(
                    nn.Linear(elem["nodes_in"], elem["nodes_out"])
                )
            elif elem["title"] == "yolo":
                res.append(
                    YoloLayer(elem["batch_size"], elem["filters"])
                )

            if elem.get("activation") == "ReLU":
                res.append(nn.ReLU())
            if elem.get("dropout"):
                res.append(nn.Dropout(elem["dropout"]))
        return res
    
    @classmethod
    def parse_config(self, config_file):
        with open(config_file, "r") as f:
            config_lines = f.readlines()
        res = []
        tmp = {}
        for line in config_lines:
            if re.search("\[[a-z]+\]", line):
                if tmp:
                    res.append(tmp)
                    tmp = {}
                layer_type = re.match("\[([a-z]+)\]", line).group(1)
                tmp['title'] = layer_type
            elif (line.strip()) and (not line.startswith("#")):
                k, v = [x.strip() for x in line.split("=")]
                if re.search("\.", v):
                    v = float(v)
                else:
                    try:
                        v = int(v)
                    except:
                        pass
                tmp[k] = v
        if tmp:
            res.append(tmp)
        return res




# %%
if __name__ == "__main__":
    model = ModelFromCFG("./mockconfig.cfg", in_ch=1)
    inp = torch.rand(1, 1, 256, 256)
    out = model(inp)
    processed_pred = process_preds(out[0], 14, 256, 56)
    t = processed_pred.detach().numpy()
    print(t.astype(np.int32)[:,:6])
# %%


# %%
