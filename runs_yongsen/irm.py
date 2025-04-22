from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import grad
from sklearn.metrics import roc_auc_score, average_precision_score
from models import SentenceBERTclf, BERTclf
from utils import mean_nll, mean_accuracy, mean_auprc, pretty_print, compute_penalty, make_environment_shac
from dataset import SHAC
import random
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colored MNIST')
    # parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--model', type=str, default='sentence-bert')
    parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_restarts', type=int, default=1)
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--grayscale', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    flags = parser.parse_args()

    print('Flags:')
    for k, v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))

    torch.cuda.set_device('cuda:3')

    shac = SHAC()

    seed = 1

    final_train_accs = []
    final_test_accs = []
    final_train_auprc = []
    final_test_auprc = []
    for restart in range(flags.n_restarts):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        seed += 1

        envs = [
            make_environment_shac(shac.train[shac.train.location == 'uw'].text,
                                  shac.train[shac.train.location == 'uw'].Drug, 0.2, flags),
            make_environment_shac(shac.train[shac.train.location == 'mimic'].text,
                                  shac.train[shac.train.location == 'mimic'].Drug, 0.1, flags),
            make_environment_shac(shac.val.text, shac.val.Drug, 0.9, flags),
            make_environment_shac(shac.test.text, shac.test.Drug, 0.9, flags),
        ]

        if flags.model == 'sentence-bert':
            clf = SentenceBERTclf().cuda()
        else:
            clf = BERTclf().cuda()

        optimizer = optim.Adam(clf.parameters(), lr=flags.lr)
        tokenizer = AutoTokenizer.from_pretrained(
            'roberta-base', model_max_length=256, truncation=True, padding=True)
        dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to("cuda")
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")

        print("Restart", restart)
        pretty_print('step', 'train nll', 'train acc',
                     'train auprc', 'train penalty', 'val acc')
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
                        encoded = tokenizer(
                            texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
                        output = clf(
                            encoded['input_ids'], attention_mask=encoded['attention_mask'])
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
                            encoded = tokenizer(
                                texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
                            output = clf(
                                encoded['input_ids'], attention_mask=encoded['attention_mask'])
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
            train_auprc = torch.cat(
                [envs[0]['auprc'], envs[1]['auprc']]).mean()
            train_penalty = torch.cat(
                [envs[0]['penalty'], envs[1]['penalty']]).mean()
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
