from typing import Optional

import cv2
import torch
from pytorch_msssim import SSIM
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score, \
    roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from torch import nn
import numpy as np
from torchmetrics import PeakSignalNoiseRatio

from .iae_loss import IAE
from ...core.alg_frame.client_trainer import ClientTrainer
import logging

from ...model.cv.cnn import CNN_OriginalFedAvg, VGG7
from ...utils.model_utils import kl_divergence_loss


class ModelTrainerCLS(ClientTrainer):

    def __init__(self, model, args):
        super().__init__(model, args)
        self.iae_algo: Optional[IAE] = None
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    def set_iae_algo(self, iae):
        self.iae_algo = iae

    def get_iae_algo(self):
        return self.iae_algo

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        self.data_size = 0
        self.lambda_l = 0

        model = self.model

        model.to(device)
        model.train()

        # train and update
        if args.dataset == "chexpert":
            ce_crit = nn.BCEWithLogitsLoss().to(device)
        else:
            ce_crit = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        re_crit = nn.MSELoss()

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        elif args.client_optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        record = {}
        epoch_loss = []
        self.embeds = []
        for epoch in range(args.epochs):
            batch_loss = []

            for batch_idx, (x, labels) in enumerate(train_data):
                if epoch == 0:
                    self.data_size += len(labels)

                x = x.to(device)
                labels = labels.float().to(device) if isinstance(ce_crit, nn.BCEWithLogitsLoss) else labels.to(
                    device).to(torch.long)
                model.zero_grad()

                if args.model == "aec":
                    emb, xp, log_probs = model(x)
                    ce_loss = ce_crit(log_probs, labels) * args.aec_ce_loss_w  # pylint: disable=E1102
                    re_loss = re_crit(xp, x) * args.aec_re_loss_w
                    loss = ce_loss + re_loss

                    if args.aec_use_vae == 1:
                        emb, mean, logvar, z = emb
                        kl_loss = kl_divergence_loss(mean, logvar) * (mean.shape[1] / np.prod(xp.shape))
                        loss += (kl_loss * args.aec_kl_loss_w)

                    if self.iae_algo is not None:
                        reps = z if args.aec_use_vae == 1 else emb
                        loss += self.iae_algo.compute_svdd_loss(reps, labels, update_r=False)
                elif isinstance(model, CNN_OriginalFedAvg) or isinstance(model, VGG7):
                    log_probs, ftrs = model(x, return_features=True)
                    if epoch == args.epochs - 1:
                        self.embeds.append(ftrs.detach())
                    loss = ce_crit(log_probs, labels)  # pylint: disable=E1102
                else:
                    log_probs = model(x)
                    loss = ce_crit(log_probs, labels)  # pylint: disable=E1102

                loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if hasattr(self.args, "gradient_clip_t") and \
                        hasattr(self.args, "gradient_clip_norm") and self.args.gradient_clip_t > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   self.args.gradient_clip_t,
                                                   self.args.gradient_clip_norm)

                if hasattr(self.args, "enable_dp") and self.args.enable_dp:
                    param_list = [p for p in model.parameters() if p.requires_grad]
                    r_item = {
                        "weights": [param.clone() for param in param_list],
                        "grads": [param.grad.clone() for param in param_list]
                    }
                    if batch_idx in record:
                        record[batch_idx].append(r_item)
                    else:
                        record[batch_idx] = [r_item]

                optimizer.step()
                batch_loss.append(loss.item())

            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if self.verbose:
                logging.info(
                    "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                        self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                    )
                )

        if len(self.embeds) > 0:
            self.embeds = torch.cat(self.embeds, dim=0)

        lambda_l_values = []
        if hasattr(self.args, "enable_dp") and self.args.enable_dp:
            for _, rec in record.items():
                for i in range(1, len(rec)):
                    prev_rec, cur_rec = rec[i - 1], rec[i]
                    g_diffs = []
                    p_diffs = []
                    for prev_g, cur_g, prev_p, cur_p in zip(prev_rec["grads"], cur_rec["grads"], prev_rec["weights"],
                                                            cur_rec["weights"]):
                        g_diffs.append((cur_g - prev_g).view(-1))
                        p_diffs.append((cur_p.data - prev_p.data).view(-1))
                    g_norm = torch.cat(g_diffs).norm(p=1).item()
                    p_norm = torch.cat(p_diffs).norm(p=1).item()
                    if p_norm > 0:
                        lambda_l = g_norm / p_norm
                        lambda_l_values.append(lambda_l)
        self.lambda_l = max(lambda_l_values) if len(lambda_l_values) > 0 else 0.0
        if self.verbose:
            logging.info("Client Index = {}\tLambda_l: {}".format(self.id, self.lambda_l))

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_re_loss": 0,
            "test_ce_loss": 0,
            "test_kl_loss": 0,
            "test_psnr": 0,
            "test_ssim": 0,
            "test_total": 0,
            'test_acc': 0,
            'test_precision': 0,
            'test_recall': 0,
            'test_bacc': 0,
            'test_auc': 0
        }

        if args.dataset == "chexpert":
            ce_crit = nn.BCEWithLogitsLoss().to(device)
        else:
            ce_crit = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        re_crit = nn.MSELoss().to(device)
        ssim = SSIM(data_range=1, win_size=7, channel=(1 if args.dataset == "mnist" else 3)).to(device)
        psnr = PeakSignalNoiseRatio(data_range=1).to(device)

        with torch.no_grad():
            correct_dict = {}
            err_dict = {}
            preds = []
            targets = []
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)

                if isinstance(ce_crit, nn.BCEWithLogitsLoss):
                    target = target.to(device).float()
                else:
                    target = target.to(device).to(torch.long)

                if args.model == "aec":
                    emb, xp, pred = model(x)
                    ce_loss = ce_crit(pred, target) * args.aec_ce_loss_w  # pylint: disable=E1102
                    re_loss = re_crit(xp, x) * args.aec_re_loss_w
                    loss = ce_loss + re_loss

                    metrics["test_re_loss"] += re_loss.item() * target.size(0)
                    metrics["test_ce_loss"] += ce_loss.item() * target.size(0)
                    metrics["test_ssim"] += ssim(xp, x).item() * target.size(0)
                    metrics["test_psnr"] += psnr(xp, x).item() * target.size(0)

                    if args.aec_use_vae == 1:
                        emb, mean, logvar, z = emb
                        kl_loss = kl_divergence_loss(mean, logvar) * (
                                mean.shape[1] / np.prod(xp.shape)) * args.aec_kl_loss_w
                        loss += kl_loss
                        metrics["test_kl_loss"] += kl_loss.item() * target.size(0)
                else:
                    pred = model(x)
                    loss = ce_crit(pred, target)  # pylint: disable=E1102

                if isinstance(ce_crit, nn.BCEWithLogitsLoss):
                    predicted = (torch.sigmoid(pred) >= 0.5).float()
                else:
                    _, predicted = torch.max(pred, -1)
                    correct = predicted.eq(target).sum()
                    for p, t in zip(predicted, target):
                        p, t = p.item(), t.item()
                        if p == t:
                            if t in correct_dict:
                                correct_dict[t] += 1
                            else:
                                correct_dict[t] = 0
                        else:
                            if t in err_dict:
                                err_dict[t] += 1
                            else:
                                err_dict[t] = 0
                    metrics["test_correct"] += correct.item()

                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)

                preds.append(predicted)
                targets.append(target)

            plist = torch.cat(preds).cpu().numpy()
            tlist = torch.cat(targets).cpu().numpy()

            if isinstance(ce_crit, nn.BCEWithLogitsLoss):
                per_label_accuracy = []
                per_label_auc = []
                for i in range(len(tlist[0])):
                    per_label_accuracy.append(accuracy_score(tlist[:, i], plist[:, i]))
                    try:
                        per_label_auc.append(roc_auc_score(tlist[:, i], plist[:, i]))
                    except Exception as ex:
                        print("Failed to compute auc per label :", ex)

                metrics['test_accuracy'] = float(np.mean(per_label_accuracy))
                metrics['test_auc'] = float(np.mean(per_label_auc))
                metrics['test_acc_per_class'] = per_label_accuracy
                metrics['test_auc_per_class'] = per_label_auc
            else:
                metrics['test_acc'] = metrics['test_correct'] / (metrics['test_total'] + 1e-13)
                acc_dict = {}
                for k in list(correct_dict.keys()) + list(err_dict.keys()):
                    e = 1e-13
                    acc_dict[k] = 0
                    if k in err_dict:
                        e += err_dict[k]
                    if k in correct_dict:
                        c = correct_dict[k]
                        acc_dict[k] = float(c / (e + c))
                metrics['test_acc_per_class'] = acc_dict
                f1s = f1_score(tlist, plist, labels=list(acc_dict.keys()), average=None)
                metrics['test_f1_per_class'] = dict(zip(list(acc_dict.keys()), f1s.tolist()))
                metrics['test_bacc'] = float(balanced_accuracy_score(tlist, plist))

            metrics['test_f1'] = float(f1_score(tlist, plist, average="macro"))
            metrics['test_precision'] = float(precision_score(tlist, plist, average="macro"))
            metrics['test_recall'] = float(recall_score(tlist, plist, average="macro"))

        return metrics

    def compute_grads(self, train_data, device, weighted=True):
        import numpy as np
        self.model = self.model.to(device)
        self.model.eval()
        if self.args.model == "aec":
            re_crit = nn.MSELoss().to(device)
        ce_crit = nn.CrossEntropyLoss().to(device)
        param_list = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        grad_norm_method = self.args.grad_method if hasattr(self.args, "grad_method") else "mean"
        if hasattr(self.args, "grad_layer_name") and len(self.args.grad_layer_name) > 0:
            new_param_list = []
            for p_name, param in self.model.named_parameters():
                if param.requires_grad and self.args.grad_layer_name in p_name:
                    new_param_list.append(param)
            if len(new_param_list) > 0:
                logging.info(f"Using {len(new_param_list)} layers for grad.")
                param_list = new_param_list
            else:
                logging.warning(f"No layers found for {self.args.grad_layer_name} so using all layers for grad.")
        grad_per_batch = []
        for batch_idx, (x, labels) in enumerate(train_data):
            batch, labels = x.to(device), labels.to(device).to(torch.long)
            grads = []
            for x, label in zip(batch, labels):
                x = x.unsqueeze(0)
                label = label.unsqueeze(0)
                if self.args.model == "aec":
                    emb, xp, pred = self.model(x)
                    ce_loss = ce_crit(pred, label)
                    re_loss = re_crit(xp, x)
                    if weighted:
                        loss = ce_loss * self.args.aec_ce_loss_w + re_loss * self.args.aec_re_loss_w
                    else:
                        loss = ce_loss + re_loss
                    if self.args.aec_use_vae == 1:
                        emb, mean, logvar, z = emb
                        kl_loss = kl_divergence_loss(mean, logvar) * (mean.shape[1] / np.prod(xp.shape))
                        if weighted:
                            loss += kl_loss * self.args.aec_kl_loss_w
                        else:
                            loss += kl_loss
                else:
                    probs = self.model(x)
                    loss = ce_crit(probs, label)
                grad = torch.autograd.grad(loss, param_list)
                if grad_norm_method == "mean":
                    gnorm_per_sample = np.mean([g.norm().item() for g in grad])
                else:
                    gnorm_per_sample = torch.stack([g.norm() for g in grad]).norm().item()
                grads.append(gnorm_per_sample)
            grad_per_batch.append(grads)
        return grad_per_batch

    def compute_loss(self, train_data, device, weighted=True, reduce='sum', return_emb=False):
        self.model = self.model.to(device)
        self.model.eval()
        ce_crit = nn.CrossEntropyLoss(reduction='none').to(device)
        if self.args.model == "aec":
            re_crit = nn.MSELoss(reduction='none').to(device)
        loss_per_batch = []
        emb_per_batch = []
        for batch_idx, (x, labels) in enumerate(train_data):
            batch, labels = x.to(device), labels.to(device).to(torch.long)

            if self.args.model == "aec":
                emb, xp, pred = self.model(batch)
                ce_loss = ce_crit(pred, labels)
                re_loss = re_crit(xp, batch).flatten(1).mean(1)
                if reduce == 'sum':
                    loss = (ce_loss * self.args.aec_ce_loss_w + re_loss * self.args.aec_re_loss_w) \
                        if weighted else (ce_loss + re_loss)
                else:
                    loss = torch.vstack([ce_loss * self.args.aec_ce_loss_w, re_loss * self.args.aec_re_loss_w]).T \
                        if weighted else torch.vstack([ce_loss, re_loss]).T
                if self.args.aec_use_vae == 1:
                    emb, mean, logvar, z = emb
                    kl_loss = kl_divergence_loss(mean, logvar, reduce=None) * (mean.shape[1] / np.prod(xp.shape[1:]))
                    if reduce == 'sum':
                        loss += kl_loss * self.args.aec_kl_loss_w if weighted else kl_loss
                    else:
                        loss = torch.cat([loss, kl_loss.unsqueeze(1) * self.args.aec_kl_loss_w], dim=1) \
                            if weighted else torch.cat([loss, kl_loss.unsqueeze(1)], dim=1)
                emb_per_batch.append(emb.cpu().detach().numpy())
            else:
                probs = self.model(batch)
                loss = ce_crit(probs, labels)
            loss_per_batch.append(loss.cpu().detach().numpy())
        if return_emb:
            return loss_per_batch, emb_per_batch
        return loss_per_batch

    def extract_features(self, train_data, device, args):
        self.model = self.model.to(device)
        self.model.eval()
        ftrs = []
        lbls = []
        for (x, labels) in train_data:
            x = x.to(device)
            if self.args.model == "aec":
                emb, xp, pred = self.model(x)
                if self.args.aec_use_vae == 1:
                    emb, mean, logvar, z = emb
            else:
                probs, emb = self.model(x, return_features=True)
            ftrs.append(emb)
            lbls.append(labels)
        return torch.cat(ftrs, dim=0), torch.cat(lbls, dim=0)
