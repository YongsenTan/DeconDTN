import torch
from torch import nn, optim
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from torch.autograd import grad
from torch.utils.data import DataLoader
from dataset import Textdataset


def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()


def mean_auprc(logits, y):
    # Detach if needed and convert to numpy
    if hasattr(logits, 'detach'):
        logits = logits.detach().cpu().numpy()
    if hasattr(y, 'detach'):
        y = y.detach().cpu().numpy()
    auprc = average_precision_score(y, logits)
    return torch.tensor(auprc).view(1)


def pretty_print(*values):
    col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


def compute_penalty(losses, dummy_w):
    g1 = grad(losses[0::2].mean(), dummy_w, create_graph=True)[0]
    g2 = grad(losses[1::2].mean(), dummy_w, create_graph=True)[0]
    return (g1 * g2).sum()


def make_environment_shac(texts, labels, e, flag):
    texts = np.array(texts)
    labels = np.array(labels, dtype=int)
    labels = torch.tensor(labels, dtype=torch.float32)

    if not flag.grayscale:
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))

    # Assign stars based on the label; flip the color with probability e
    stars = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the stars to the texts
    text_stars = []
    modified_cnt = 0
    for star, text in zip(stars, texts):
        # UW deidentification
        if star == 0 and not flag.grayscale:
            modified_cnt += 1
            if '[**' in text:
                text = text.replace('[**', '[').replace('**]', ']')
            # else:
            #   text = text.replace('[', '[**').replace(']', '**]')
        text_stars.append(text)

    # print("Modified %d texts" % modified_cnt)

    dataset = Textdataset(text_stars, labels)
    dataloader = DataLoader(dataset, flag.batch_size, shuffle=True)
    return {
        'texts': text_stars,
        'labels': labels[:, None].cuda(),
        'dataloader': dataloader
    }


def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()


def torch_xor(a, b):
    return (a-b).abs()  # Assumes both inputs are either 0 or 1
