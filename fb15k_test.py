from dgl.data import FB15k237Dataset
from model import LinkPredictionModel, PretrainedGraphEncoder, MLP, CLIP_KB, GPT2CaptionEncoder, BertCaptionEncoder, ConcatModel, RGCN
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from utils import training_routine, KG
import torch


dataset = FB15k237Dataset(reverse=False)
graph = dataset[0]
train_mask = graph.edata['train_mask']
train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
src, dst = graph.find_edges(train_idx)
rel = graph.edata['etype'][train_idx]
print(src.shape)
