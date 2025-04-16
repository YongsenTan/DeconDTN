import sys
sys.path.append('./DeconDTN-Toolkit')

from DeconT.preprocessing import load_process_SHAC
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np
import torch
# from torchvision import datasets
from torch import nn, optim, autograd
from torch.autograd import grad 
from sklearn.metrics import roc_auc_score, average_precision_score


# Build environments for the colored MNIST task
def make_environment(images, labels, e):
  def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()
  def torch_xor(a, b):
    return (a-b).abs() # Assumes both inputs are either 0 or 1
  # 2x subsample for computational convenience
  images = images.reshape((-1, 28, 28))[:, ::2, ::2]
  # Assign a binary label based on the digit; flip label with probability 0.25
  labels = (labels < 5).float()
  labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
  # Assign a color based on the label; flip the color with probability e
  colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
  # Apply the color to the image by zeroing out the other color channel
  images = torch.stack([images, images], dim=1)
  images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
  return {
    'images': (images.float() / 255.).cuda(),
    'labels': labels[:, None].cuda()
  }
  
class Textdataset(Dataset):
  def __init__(self, texts, labels):
    self.texts = texts
    self.labels = labels[:, None]

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    return self.texts[idx], self.labels[idx]
  
  
# Build environments for the SHAC task
# Build a 'provenance-mnist' dataset: different deidentifying methods
def make_environment_shac(texts, labels, e, flag):
  def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()
  def torch_xor(a, b):
    return (a-b).abs() # Assumes both inputs are either 0 or 1
  
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
      # UW
      if star == 0 and not flag.grayscale:
        modified_cnt += 1
        if '[**' in text:
          text = text.replace('[**', '[').replace('**]', ']')
        # else:
        #   text = text.replace('[', '[**').replace(']', '**]')
      text_stars.append(text)
  
  print("Modified %d texts" % modified_cnt)
  
  dataset = Textdataset(texts=text_stars, labels=labels)
  dataloader = DataLoader(dataset, flag.batch_size, shuffle=True)
  return {
  'texts': text_stars,
  'labels': labels[:, None].cuda(),
  'dataloader': dataloader
  }


class SentenceBERTclf(nn.Module):
  def __init__(self):
    super(SentenceBERTclf, self).__init__()
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    # for param in self.model.parameters():
    #   param.requires_grad = False
    self.classifier = nn.Sequential(nn.Linear(384, 256),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(256, 1))

  def forward(self, input):
    out = self.model.encode(input)
    out = torch.tensor(out).cuda()
    out = self.classifier(out)
    return out
  
  
class BERTclf(nn.Module):
    def __init__(self, num_labels=1, dropout=0.1,bert_model='roberta-base'):
        super(BERTclf, self).__init__()
        self.model = AutoModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        dropped = self.dropout(pooled_output)
        logits = self.classifier(dropped)
        return logits


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


# def penalty(logits, y):
#   scale = torch.tensor(1.).cuda().requires_grad_()
#   loss = mean_nll(logits * scale, y)
#   grad = autograd.grad(loss, [scale], create_graph=True)[0]
#   return torch.sum(grad**2)

def compute_penalty(losses , dummy_w): 
  g1 = grad(losses[0::2].mean(), dummy_w, create_graph=True)[0] 
  g2 = grad(losses[1::2].mean(), dummy_w, create_graph=True)[0] 
  return (g1 * g2).sum()


def pretty_print(*values):
  col_width = 13
  def format_val(v):
    if not isinstance(v, str):
      v = np.array2string(v, precision=5, floatmode='fixed')
    return v.ljust(col_width)
  str_values = [format_val(v) for v in values]
  print("   ".join(str_values))

  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Colored MNIST')
  # parser.add_argument('--hidden_dim', type=int, default=256)
  parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--n_restarts', type=int, default=1)
  parser.add_argument('--penalty_anneal_iters', type=int, default=100)
  parser.add_argument('--penalty_weight', type=float, default=10000.0)
  parser.add_argument('--steps', type=int, default=20)
  parser.add_argument('--grayscale', type=int, default=0)
  parser.add_argument('--batch_size', type=int, default=128)
  flags = parser.parse_args()
  
  print('Flags:')
  for k,v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))
  
  torch.cuda.set_device('cuda:3')
  
  shac = load_process_SHAC(replaceNA='all', base_dir='/edata/xiruod/SocialDeterminants_SHAC_n2c2_2022/n2c2_sdoh_challenge')
  
  # print(shac.head())

  shac_train, shac_val, shac_test = shac[shac.set =='train'], shac[shac.set =='dev'], shac[shac.set =='test']
  print("Training: %d, Validation: %d, Test: %d" % (len(shac_train), len(shac_val), len(shac_test)))
  

  envs = [
    make_environment_shac(shac_train[shac_train.location == 'uw'].text, shac_train[shac_train.location == 'uw'].Drug, 0.2, flags),
    make_environment_shac(shac_train[shac_train.location == 'mimic'].text, shac_train[shac_train.location == 'mimic'].Drug, 0.1, flags),
    make_environment_shac(shac_val.text, shac_val.Drug, 0.9, flags),
    make_environment_shac(shac_test.text, shac_test.Drug, 0.9, flags),
  ]
  

  final_train_accs = []
  final_test_accs = []
  final_train_auprc = []
  final_test_auprc = []
  for restart in range(flags.n_restarts):
    clf = SentenceBERTclf().cuda()
    # clf = BERTclf().cuda()
    optimizer = optim.Adam(clf.parameters(), lr=flags.lr)
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', model_max_length=256, truncation=True, padding=True)
    dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to("cuda")
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    
    print("Restart", restart)
    pretty_print('step', 'train nll', 'train acc', 'train auprc', 'train penalty', 'val acc')
    val_best = -1
    for step in range(flags.steps):
      for env in envs[:2]:
        logits, label_list, nll_list, acc_list, auprc_list, penalty_list = [], [], [], [], [], []
        for texts, labels in env['dataloader']:
          error = 0 
          penalty = 0
          label_list.append(labels)
          labels = labels.to("cuda")
          if isinstance(clf, SentenceBERTclf):
            output = clf(texts)
          else:
            encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
            output = clf(encoded['input_ids'], attention_mask=encoded['attention_mask'])
          logits.append(output)
          
          nll_batch = bce(output * dummy_w, labels)
          penalty += compute_penalty(nll_batch, dummy_w)
          error += nll_batch.mean()

          nll_list.append(nll_batch)
          penalty_list.append(penalty.view(1))
          accuracy_batch = mean_accuracy(output, labels)
          acc_list.append(accuracy_batch.view(1))
          auprc_batch = mean_auprc(output, labels)
          auprc_list.append(auprc_batch)

          # L2 normalization on the MLP
          weight_norm = torch.tensor(0.).cuda()
          for w in clf.classifier.parameters():
            weight_norm += w.norm().pow(2)
          error += flags.l2_regularizer_weight * weight_norm
          
          # W penalty
          penalty_weight = (flags.penalty_weight 
              if step >= flags.penalty_anneal_iters else 1.0)
          error += penalty_weight * penalty
          if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            error /= penalty_weight

          optimizer.zero_grad()
          error.backward()
          optimizer.step()
        
        logits = torch.cat(logits, dim=0).detach().cpu()
        label_list = torch.cat(label_list, dim=0)
        env['nll'] = torch.cat(nll_list, dim=0)
        env['acc'] = torch.cat(acc_list, dim=0)
        env['penalty'] = torch.cat(penalty_list, dim=0)
        env['auprc'] = torch.cat(auprc_list, dim=0)
      
      for e_i, env in enumerate(envs[-2:]):
        with torch.no_grad():
          logits, nll_list, acc_list, auprc_list, penalty_list = [], [], [], [], []
          for texts, labels in envs[-2+e_i]['dataloader']:
            labels = labels.to("cuda")
            if isinstance(clf, SentenceBERTclf):
              output = clf(texts)
            else:
              encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
              output = clf(encoded['input_ids'], attention_mask=encoded['attention_mask'])
            logits.append(output)
            
            accuracy_batch = mean_accuracy(output, labels)
            acc_list.append(accuracy_batch.view(1))
            auprc_batch = mean_auprc(output, labels)
            auprc_list.append(auprc_batch)

          logits = torch.cat(logits, dim=0)
          envs[-2+e_i]['acc'] = torch.cat(acc_list, dim=0)
          envs[-2+e_i]['auprc'] = torch.cat(auprc_list, dim=0)

        
        
      train_nll = torch.cat([envs[0]['nll'], envs[1]['nll']]).mean()
      train_acc = torch.cat([envs[0]['acc'], envs[1]['acc']]).mean()
      train_auprc = torch.cat([envs[0]['auprc'], envs[1]['auprc']]).mean()
      train_penalty = torch.cat([envs[0]['penalty'], envs[1]['penalty']]).mean()
      val_acc = envs[-2]['acc'].mean()
      if val_acc > val_best:
        val_best = val_acc
        test_acc = envs[-1]['acc'].mean()
        test_auprc = envs[-1]['auprc'].mean()
      
      pretty_print(
        np.int32(step),
        train_nll.detach().cpu().numpy(),
        train_acc.detach().cpu().numpy(),
        train_auprc,
        train_penalty.detach().cpu().numpy(),
        val_acc.detach().cpu().numpy()
      )

    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
    
    final_train_auprc.append(train_auprc.detach().cpu().numpy())
    final_test_auprc.append(test_auprc.detach().cpu().numpy())
    print('Final train auprc (mean/std across restarts so far):')
    print(np.mean(final_train_auprc), np.std(final_train_auprc))
    print('Final test auprc (mean/std across restarts so far):')
    print(np.mean(final_test_auprc), np.std(final_test_auprc))
    