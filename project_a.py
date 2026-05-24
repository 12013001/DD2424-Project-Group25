import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.metrics import f1_score, confusion_matrix

from project_e import (
    PrepareData, BuildBackbone, UnfreezeLastL, FreezeAllExceptFC, SetRequiresGrad,
    TrainModel, TrainOneEpoch, Evaluate, EvaluateTTA, MakeLoader,
    StratifiedFraction, GetLabels,
    PlotHistory, PlotPerClassF1, AppendCSV, ResetCSV,
    DEVICE, BATCH_SIZE, NUM_WORKERS, RESNET_LAYER_GROUPS,
)

RESULTS_DIR_A = "./results_a"
LABELED_FRACTIONS = [1.0, 0.5, 0.1, 0.01]
DEFAULT_THRESHOLD = 0.9
THRESHOLD_VALUES = [0.7, 0.8, 0.9, 0.95]
ABLATION_FRACTION = 0.1
STRATEGY_PER_FRAC = {1.0: "s1", 0.5: "s1", 0.1: "s2", 0.01: "s2"}
SUMMARY_HEADER_A = ["experiment", "config", "best_val_acc", "test_acc", "macro_f1", "weighted_f1", "train_time_sec"]

class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, labels):
        self.base = base_dataset
        self.indices = indices
        self.labels = labels
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        img, _ = self.base[self.indices[i]]
        return img, self.labels[i]

def MakeCombinedLoader(data, labeled_idx, labeled_labels, pseudo_idx, pseudo_labels):
    labeled_ds = LabeledDataset(data["trainval_aug"], labeled_idx, labeled_labels)
    if pseudo_idx:
        pseudo_ds = LabeledDataset(data["trainval_aug"], pseudo_idx, pseudo_labels)
        combined = ConcatDataset([labeled_ds, pseudo_ds])
    else:
        combined = labeled_ds
    return MakeLoader(combined)

def GeneratePseudoLabels(teacher, data, unlabeled_idx, threshold):
    if not unlabeled_idx:
        return [], [], 0, 0, 0.0, None
    loader = DataLoader(Subset(data["trainval_clean"], unlabeled_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda",)
    teacher.eval()
    all_max_probs, all_preds, all_true = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            probs = torch.softmax(teacher(x), dim=1)
            max_probs, preds = probs.max(dim=1)
            all_max_probs.extend(max_probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(y.cpu().tolist())
    pseudo_idx_out, pseudo_labels_out, pseudo_true_out = [], [], []
    for orig_idx, prob, pred, true_lbl in zip(unlabeled_idx, all_max_probs, all_preds, all_true):
        if prob >= threshold:
            pseudo_idx_out.append(orig_idx)
            pseudo_labels_out.append(pred)
            pseudo_true_out.append(true_lbl)
    n_total = len(unlabeled_idx)
    n_accepted = len(pseudo_idx_out)
    accept_rate = n_accepted / n_total if n_total > 0 else 0.0
    pseudo_acc = (float(np.mean(np.array(pseudo_labels_out) == np.array(pseudo_true_out)))
                   if n_accepted > 0 else None)
    return pseudo_idx_out, pseudo_labels_out, n_total, n_accepted, accept_rate, pseudo_acc

def TrainAndEvaluateS1(data, backbone, best_l, train_loader, n_epochs):
    val_loader = MakeLoader(data["val_mc"],  shuffle=False)
    test_loader = MakeLoader(data["test_mc"], shuffle=False)
    model = BuildBackbone(backbone, 37, pretrained=True)
    UnfreezeLastL(model, best_l)
    model = model.to(DEVICE)
    t_start = time.time()
    model, history, best_val = TrainModel(model, train_loader, val_loader, n_epochs=n_epochs, lr=1e-4, wd=1e-3, opt_type="adamw", label_smoothing=0.1,)
    train_time = time.time() - t_start
    _, test_acc, preds, lbls = EvaluateTTA(model, test_loader, nn.CrossEntropyLoss())
    macro_f1 = f1_score(lbls, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(lbls, preds, average="weighted", zero_division=0)
    return {"model": model, "history": history, "best_val": best_val,
            "test_acc": test_acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "train_time": train_time, "preds": preds, "lbls": lbls}

def TrainAndEvaluateS2(data, backbone, train_loader, n_epochs_per_stage):
    val_loader = MakeLoader(data["val_mc"],  shuffle=False)
    test_loader = MakeLoader(data["test_mc"], shuffle=False)
    model = BuildBackbone(backbone, 37, pretrained=True)
    FreezeAllExceptFC(model)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs_per_stage)
    best_val = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    t_start = time.time()
    for stage in range(len(RESNET_LAYER_GROUPS) + 1):
        for _ in range(n_epochs_per_stage):
            tr_loss, tr_acc = TrainOneEpoch(model, train_loader, criterion, optimizer)
            vl_loss, vl_acc, _, _ = Evaluate(model, val_loader, criterion)
            scheduler.step()
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(vl_loss)
            history["val_acc"].append(vl_acc)
            if vl_acc > best_val:
                best_val = vl_acc
                best_wts = copy.deepcopy(model.state_dict())
        if stage < len(RESNET_LAYER_GROUPS):
            g = RESNET_LAYER_GROUPS[stage]
            if g == "conv1_bn1":
                SetRequiresGrad(model.conv1, True)
                SetRequiresGrad(model.bn1,   True)
            else:
                SetRequiresGrad(getattr(model, g), True)
            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-4, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs_per_stage)
    train_time = time.time() - t_start
    model.load_state_dict(best_wts)
    _, test_acc, preds, lbls = EvaluateTTA(model, test_loader, nn.CrossEntropyLoss())
    macro_f1 = f1_score(lbls, preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(lbls, preds, average="weighted", zero_division=0)
    return {"model": model, "history": history, "best_val": best_val,
            "test_acc": test_acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "train_time": train_time, "preds": preds, "lbls": lbls}

# Part A1
def RunSupervisedBaselines(data, backbone, best_l, n_epochs, n_epochs_per_stage, out_dir):
    all_labels = GetLabels(data["trainval_clean"])
    ResetCSV(os.path.join(out_dir, "supervised_baseline.csv"))
    results = {}
    for frac in LABELED_FRACTIONS:
        strategy = STRATEGY_PER_FRAC.get(frac, "s1")
        labeled_idx = StratifiedFraction(data["train_indices"], data["train_labels"], frac)
        labeled_labels = [all_labels[i] for i in labeled_idx]
        train_loader = MakeLoader(Subset(data["trainval_aug"], labeled_idx))
        if strategy == "s2":
            r = TrainAndEvaluateS2(data, backbone, train_loader, n_epochs_per_stage)
            tag = (f"backbone={backbone}, strategy=gradual_unfreezing, fraction={frac}, "
                   f"n_labeled={len(labeled_idx)}, optimizer=AdamW, "
                   f"n_epochs_per_stage={n_epochs_per_stage}, wd=1e-3, LS=0.1, TTA=yes")
        else:
            r = TrainAndEvaluateS1(data, backbone, best_l, train_loader, n_epochs)
            tag = (f"backbone={backbone}, strategy=s1, l={best_l}, fraction={frac}, "
                   f"n_labeled={len(labeled_idx)}, optimizer=AdamW, "
                   f"eta=1e-4, wd=1e-3, LS=0.1, n_epochs={n_epochs}, TTA=yes")
        r["labeled_idx"] = labeled_idx
        r["labeled_labels"] = labeled_labels
        r["strategy"] = strategy
        results[frac] = r
        print(tag)
        print(f"best_val_acc={r['best_val']:.4f}")
        print(f"test_accuracy={r['test_acc']:.4f}")
        print(f"macro_F1={r['macro_f1']:.4f}")
        print(f"weighted_F1={r['weighted_f1']:.4f}")
        print(f"train_time={r['train_time']:.1f}s")
        print()
        PlotHistory(r["history"], f"PartA1 supervised fraction={int(frac*100)}%", save_path=os.path.join(out_dir, f"PartA1_sup_frac{int(frac*100)}pct.png"))
        AppendCSV(os.path.join(out_dir, "supervised_baseline.csv"),
            SUMMARY_HEADER_A,
            [f"PartA1_sup_frac{frac}", tag,
             r["best_val"], r["test_acc"], r["macro_f1"], r["weighted_f1"],
             round(r["train_time"], 1)])
    return results

# Part A2
def RunPseudoLabeling(data, backbone, best_l, n_epochs, n_epochs_per_stage, threshold, out_dir, supervised_results=None):
    all_labels = GetLabels(data["trainval_clean"])
    ResetCSV(os.path.join(out_dir, "pseudo_labeling.csv"))
    header = ["fraction", "threshold", "strategy",
              "n_labeled", "n_unlabeled", "n_pseudo", "accept_rate",
              "pseudo_label_acc",
              "teacher_test_acc", "student_test_acc",
              "teacher_macro_f1", "student_macro_f1",
              "delta_acc", "delta_macro_f1", "train_time_sec"]
    results = {}
    for frac in LABELED_FRACTIONS:
        strategy = STRATEGY_PER_FRAC.get(frac, "s1")
        if supervised_results and frac in supervised_results:
            sv = supervised_results[frac]
            teacher = sv["model"]
            teacher_acc = sv["test_acc"]
            teacher_f1 = sv["macro_f1"]
            labeled_idx = sv["labeled_idx"]
            labeled_labels = sv["labeled_labels"]
        else:
            labeled_idx = StratifiedFraction(data["train_indices"], data["train_labels"], frac)
            labeled_labels = [all_labels[i] for i in labeled_idx]
            train_loader = MakeLoader(Subset(data["trainval_aug"], labeled_idx))
            if strategy == "s2":
                sv = TrainAndEvaluateS2(data, backbone, train_loader, n_epochs_per_stage)
            else:
                sv = TrainAndEvaluateS1(data, backbone, best_l, train_loader, n_epochs)
            teacher = sv["model"]
            teacher_acc = sv["test_acc"]
            teacher_f1 = sv["macro_f1"]
        labeled_set = set(labeled_idx)
        unlabeled_idx = [i for i in data["train_indices"] if i not in labeled_set]
        pseudo_idx, pseudo_labels, n_total, n_accepted, accept_rate, pseudo_acc = \
            GeneratePseudoLabels(teacher, data, unlabeled_idx, threshold)
        pseudo_acc_str = f"{pseudo_acc:.4f}" if pseudo_acc is not None else "NA"
        if not unlabeled_idx:
            print(f"fraction=100%: no unlabeled data, student=teacher")
            print(f"teacher_test_accuracy={teacher_acc:.4f}")
            print(f"teacher_macro_F1={teacher_f1:.4f}")
            print()
            results[frac] = {
                "student_test_acc": teacher_acc, "student_macro_f1": teacher_f1,
                "teacher_test_acc": teacher_acc, "teacher_macro_f1": teacher_f1,
                "delta_acc": 0.0, "delta_macro_f1": 0.0,
                "n_labeled": len(labeled_idx), "n_pseudo": 0,
                "accept_rate": 0.0, "pseudo_acc": None,
            }
            AppendCSV(os.path.join(out_dir, "pseudo_labeling.csv"), header,
                [frac, threshold, strategy, len(labeled_idx), 0,
                 0, 0.0, "NA",
                 teacher_acc, teacher_acc,
                 teacher_f1, teacher_f1,
                 0.0, 0.0, 0.0])
            continue
        combined_loader = MakeCombinedLoader(data, labeled_idx, labeled_labels, pseudo_idx, pseudo_labels)
        if strategy == "s2":
            r = TrainAndEvaluateS2(data, backbone, combined_loader, n_epochs_per_stage)
        else:
            r = TrainAndEvaluateS1(data, backbone, best_l, combined_loader, n_epochs)
        delta_acc = r["test_acc"]  - teacher_acc
        delta_f1 = r["macro_f1"] - teacher_f1
        results[frac] = {
            "student_test_acc": r["test_acc"],
            "student_macro_f1": r["macro_f1"],
            "student_weighted_f1": r["weighted_f1"],
            "teacher_test_acc": teacher_acc,
            "teacher_macro_f1": teacher_f1,
            "delta_acc": delta_acc,
            "delta_macro_f1": delta_f1,
            "n_labeled": len(labeled_idx),
            "n_pseudo": n_accepted,
            "accept_rate": accept_rate,
            "pseudo_acc": pseudo_acc,
            "history": r["history"],
            "preds": r["preds"],
            "lbls": r["lbls"],
        }
        epoch_desc = (f"n_epochs_per_stage={n_epochs_per_stage}, total_epochs={6 * n_epochs_per_stage}"
                      if strategy == "s2" else f"n_epochs={n_epochs}")
        tag = (f"backbone={backbone}, strategy={strategy}, fraction={frac}, "
               f"threshold={threshold}, "
               f"n_labeled={len(labeled_idx)}, n_pseudo={n_accepted}/{n_total}, "
               f"{epoch_desc}, TTA=yes")
        print(tag)
        print(f"pseudo_label_accuracy={pseudo_acc_str}")
        print(f"teacher_test_accuracy={teacher_acc:.4f}")
        print(f"student_test_accuracy={r['test_acc']:.4f}")
        print(f"delta_accuracy={delta_acc:+.4f}")
        print(f"teacher_macro_F1={teacher_f1:.4f}")
        print(f"student_macro_F1={r['macro_f1']:.4f}")
        print(f"delta_macro_F1={delta_f1:+.4f}")
        print(f"accept_rate={accept_rate:.1%}")
        print(f"train_time={r['train_time']:.1f}s")
        print()
        PlotHistory(r["history"],
                    f"PartA2 pseudo fraction={int(frac*100)}% thr={threshold}",
                    save_path=os.path.join(out_dir,
                        f"PartA2_pseudo_frac{int(frac*100)}pct_thr{int(threshold*100)}.png"))
        AppendCSV(os.path.join(out_dir, "pseudo_labeling.csv"), header,
            [frac, threshold, strategy, len(labeled_idx), len(unlabeled_idx),
             n_accepted, round(accept_rate, 4), pseudo_acc_str,
             teacher_acc, r["test_acc"],
             teacher_f1,  r["macro_f1"],
             delta_acc, delta_f1, round(r["train_time"], 1)])
    return results

# Part A3
def RunThresholdAblation(data, backbone, best_l, n_epochs, n_epochs_per_stage, out_dir):
    fraction = ABLATION_FRACTION
    strategy = STRATEGY_PER_FRAC.get(fraction, "s1")
    all_labels = GetLabels(data["trainval_clean"])
    labeled_idx = StratifiedFraction(data["train_indices"], data["train_labels"], fraction)
    labeled_labels = [all_labels[i] for i in labeled_idx]
    labeled_set = set(labeled_idx)
    unlabeled_idx = [i for i in data["train_indices"] if i not in labeled_set]
    train_loader = MakeLoader(Subset(data["trainval_aug"], labeled_idx))
    print(f"threshold ablation: training teacher ({strategy}) at fraction={int(fraction*100)}%, n_labeled={len(labeled_idx)}")
    if strategy == "s2":
        sv = TrainAndEvaluateS2(data, backbone, train_loader, n_epochs_per_stage)
    else:
        sv = TrainAndEvaluateS1(data, backbone, best_l, train_loader, n_epochs)
    teacher = sv["model"]
    print(f"teacher_test_accuracy={sv['test_acc']:.4f}")
    print(f"teacher_macro_F1={sv['macro_f1']:.4f}")
    print()
    header = ["fraction", "threshold", "n_labeled", "n_pseudo", "accept_rate",
              "pseudo_label_acc", "best_val_acc",
              "test_acc", "macro_f1", "delta_acc", "delta_macro_f1", "train_time_sec"]
    ResetCSV(os.path.join(out_dir, "threshold_ablation.csv"))
    results = {"teacher": {"test_acc": sv["test_acc"], "macro_f1": sv["macro_f1"],
                            "best_val": sv["best_val"]}}
    for thr in THRESHOLD_VALUES:
        pseudo_idx, pseudo_labels, n_total, n_accepted, accept_rate, pseudo_acc = \
            GeneratePseudoLabels(teacher, data, unlabeled_idx, thr)
        pseudo_acc_str = f"{pseudo_acc:.4f}" if pseudo_acc is not None else "NA"
        combined_loader = MakeCombinedLoader(
            data, labeled_idx, labeled_labels, pseudo_idx, pseudo_labels)
        if strategy == "s2":
            r = TrainAndEvaluateS2(data, backbone, combined_loader, n_epochs_per_stage)
        else:
            r = TrainAndEvaluateS1(data, backbone, best_l, combined_loader, n_epochs)
        delta_acc = r["test_acc"]  - sv["test_acc"]
        delta_f1  = r["macro_f1"] - sv["macro_f1"]
        results[thr] = {
            "test_acc": r["test_acc"],
            "best_val": r["best_val"],
            "macro_f1": r["macro_f1"],
            "n_pseudo": n_accepted,
            "accept_rate": accept_rate,
            "pseudo_acc": pseudo_acc,
            "delta_acc": delta_acc,
            "delta_f1": delta_f1,
        }
        epoch_desc = (f"n_epochs_per_stage={n_epochs_per_stage}, total_epochs={6 * n_epochs_per_stage}"
                      if strategy == "s2" else f"n_epochs={n_epochs}")
        tag = (f"backbone={backbone}, strategy={strategy}, "
               f"fraction={fraction}, threshold={thr}, "
               f"n_labeled={len(labeled_idx)}, n_pseudo={n_accepted}/{n_total}, "
               f"{epoch_desc}, TTA=yes")
        print(tag)
        print(f"pseudo_label_accuracy={pseudo_acc_str}")
        print(f"best_val_acc={r['best_val']:.4f}")
        print(f"test_accuracy={r['test_acc']:.4f}")
        print(f"delta_accuracy={delta_acc:+.4f}")
        print(f"macro_F1={r['macro_f1']:.4f}")
        print(f"delta_macro_F1={delta_f1:+.4f}")
        print(f"accept_rate={accept_rate:.1%}")
        print(f"train_time={r['train_time']:.1f}s")
        print()
        AppendCSV(os.path.join(out_dir, "threshold_ablation.csv"), header,
            [fraction, thr, len(labeled_idx), n_accepted,
             round(accept_rate, 4), pseudo_acc_str, r["best_val"],
             r["test_acc"], r["macro_f1"],
             delta_acc, delta_f1, round(r["train_time"], 1)])
    return results

def PlotConfusionMatrix(lbls, preds, class_names, title, save_path=None):
    cm = confusion_matrix(lbls, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=5)
    ax.set_yticklabels(class_names, fontsize=5)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.close()

def PrintTopConfusedPairs(lbls, preds, class_names, n=10):
    cm = confusion_matrix(lbls, preds, labels=list(range(len(class_names))))
    np.fill_diagonal(cm, 0)
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    print(f"top {n} confused pairs (true -> predicted):")
    for count, true_cls, pred_cls in pairs[:n]:
        print(f"  {true_cls} -> {pred_cls}: {count}")

def PlotComparison(sup_results, pseudo_results, out_dir):
    fracs = sorted(LABELED_FRACTIONS, reverse=True)
    labels = [f"{int(f*100)}%" for f in fracs]
    sup_acc = [sup_results[f]["test_acc"] for f in fracs]
    sup_f1 = [sup_results[f]["macro_f1"] for f in fracs]
    ps_acc = [pseudo_results[f].get("student_test_acc", sup_results[f]["test_acc"])
               for f in fracs]
    ps_f1 = [pseudo_results[f].get("student_macro_f1", sup_results[f]["macro_f1"])
               for f in fracs]
    x = np.arange(len(fracs))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, sv_vals, ps_vals, ylabel, title in [
        (ax1, sup_acc, ps_acc, "test accuracy",
         "Test Accuracy: Supervised vs Pseudo-labeling"),
        (ax2, sup_f1,  ps_f1,  "macro F1",
         "Macro F1: Supervised vs Pseudo-labeling"),
    ]:
        b1 = ax.bar(x - 0.2, sv_vals, 0.4, label="supervised",
                    color="steelblue", alpha=0.75, edgecolor="white")
        b2 = ax.bar(x + 0.2, ps_vals, 0.4, label="pseudo-labeling",
                    color="tomato",    alpha=0.75, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, min(1.05, max(sv_vals + ps_vals) + 0.12))
        ax.legend()
        for b, v in list(zip(b1, sv_vals)) + list(zip(b2, ps_vals)):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "PartA4_comparison_sup_vs_pseudo.png"), dpi=150)
    plt.close()

def PlotThresholdAblation(ablation_results, out_dir):
    fraction = ABLATION_FRACTION
    thresholds = sorted(k for k in ablation_results if k != "teacher")
    accept_pct = [ablation_results[t]["accept_rate"] * 100 for t in thresholds]
    ps_acc_vals = [ablation_results[t]["pseudo_acc"]        for t in thresholds]
    test_accs = [ablation_results[t]["test_acc"]          for t in thresholds]
    delta_pp = [ablation_results[t]["delta_acc"] * 100   for t in thresholds]
    teacher_acc = ablation_results["teacher"]["test_acc"]
    thr_labels = [str(t) for t in thresholds]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4))
    ax1.bar(thr_labels, accept_pct, color="steelblue", alpha=0.75, edgecolor="white")
    for i, v in enumerate(accept_pct):
        ax1.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax1.set_xlabel("confidence threshold")
    ax1.set_ylabel("accept rate (%)")
    ax1.set_title("pseudo-label accept rate")
    ps_acc_plot = [v if v is not None else 0.0 for v in ps_acc_vals]
    ax2.bar(thr_labels, [v * 100 for v in ps_acc_plot],
            color="darkorange", alpha=0.75, edgecolor="white")
    for i, v in enumerate(ps_acc_vals):
        label = f"{v*100:.1f}%" if v is not None else "NA"
        ax2.text(i, (v * 100 if v else 0) + 0.5, label, ha="center", fontsize=9)
    ax2.set_xlabel("confidence threshold")
    ax2.set_ylabel("pseudo-label accuracy (%)")
    ax2.set_title("pseudo-label accuracy")
    ax3.plot(thr_labels, test_accs, "o-", color="tomato", linewidth=2, markersize=7)
    ax3.axhline(teacher_acc, color="gray", linestyle="--",
                label=f"teacher baseline ({teacher_acc:.3f})")
    for i, v in enumerate(test_accs):
        ax3.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    ax3.set_xlabel("confidence threshold")
    ax3.set_ylabel("test accuracy")
    ax3.set_title("test accuracy vs threshold")
    ax3.legend()
    colors = ["tomato" if d >= 0 else "steelblue" for d in delta_pp]
    ax4.bar(thr_labels, delta_pp, color=colors, alpha=0.75, edgecolor="white")
    ax4.axhline(0, color="black", linewidth=0.8)
    for i, v in enumerate(delta_pp):
        ax4.text(i, v + (0.1 if v >= 0 else -0.3), f"{v:+.2f}", ha="center", fontsize=9)
    ax4.set_xlabel("confidence threshold")
    ax4.set_ylabel("delta accuracy (pp)")
    ax4.set_title("accuracy gain over teacher")
    plt.suptitle(f"PartA3 threshold ablation at {int(fraction*100)}% labeled data")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "PartA3_threshold_ablation.png"), dpi=150)
    plt.close()

# Part A4
def RunFailureAnalysis(data, sup_results, pseudo_results, out_dir):
    frac = ABLATION_FRACTION
    class_names = data["class_names"]
    cat_indices = data["cat_indices"]
    dog_indices = data["dog_indices"]
    if frac not in sup_results or frac not in pseudo_results:
        print(f"failure analysis skipped: results missing for fraction={frac}")
        return
    sv = sup_results[frac]
    ps = pseudo_results[frac]
    PlotPerClassF1(sv["lbls"], sv["preds"], class_names,
                   f"PartA4 supervised {int(frac*100)}% labeled per-class F1",
                   highlight_indices=cat_indices,
                   save_path=os.path.join(out_dir,
                       f"PartA4_sup_{int(frac*100)}pct_per_class_F1.png"))
    PlotConfusionMatrix(sv["lbls"], sv["preds"], class_names,
                        f"PartA4 supervised {int(frac*100)}% confusion matrix",
                        save_path=os.path.join(out_dir,
                            f"PartA4_sup_{int(frac*100)}pct_confusion_matrix.png"))
    print(f"supervised confusion matrix ({int(frac*100)}% labeled):")
    PrintTopConfusedPairs(sv["lbls"], sv["preds"], class_names, n=10)
    print()
    if "preds" in ps:
        PlotPerClassF1(ps["lbls"], ps["preds"], class_names,
                       f"PartA4 pseudo-labeling {int(frac*100)}% per-class F1",
                       highlight_indices=cat_indices,
                       save_path=os.path.join(out_dir,
                           f"PartA4_pseudo_{int(frac*100)}pct_per_class_F1.png"))
        PlotConfusionMatrix(ps["lbls"], ps["preds"], class_names,
                            f"PartA4 pseudo-labeling {int(frac*100)}% confusion matrix",
                            save_path=os.path.join(out_dir,
                                f"PartA4_pseudo_{int(frac*100)}pct_confusion_matrix.png"))
        print(f"pseudo-labeling confusion matrix ({int(frac*100)}% labeled):")
        PrintTopConfusedPairs(ps["lbls"], ps["preds"], class_names, n=10)
        print()
    def GroupF1(preds, lbls):
        cat_lbls = [l for l in lbls  if l in cat_indices]
        cat_preds = [p for p, l in zip(preds, lbls) if l in cat_indices]
        dog_lbls = [l for l in lbls  if l in dog_indices]
        dog_preds = [p for p, l in zip(preds, lbls) if l in dog_indices]
        cat_f1 = f1_score(cat_lbls, cat_preds,
                          labels=sorted(cat_indices), average="macro", zero_division=0)
        dog_f1 = f1_score(dog_lbls, dog_preds,
                          labels=sorted(dog_indices), average="macro", zero_division=0)
        return cat_f1, dog_f1
    sv_cat, sv_dog = GroupF1(sv["preds"], sv["lbls"])
    print(f"failure analysis at fraction={int(frac*100)}%")
    print(f"supervised: test_accuracy={sv['test_acc']:.4f}, macro_F1={sv['macro_f1']:.4f}, cat_F1={sv_cat:.4f}, dog_F1={sv_dog:.4f}")
    if "preds" in ps:
        ps_cat, ps_dog = GroupF1(ps["preds"], ps["lbls"])
        pa = ps.get("pseudo_acc")
        pa_str = f"{pa:.4f}" if pa is not None else "NA"
        print(f"pseudo-labeling: test_accuracy={ps['student_test_acc']:.4f}, macro_F1={ps['student_macro_f1']:.4f}, cat_F1={ps_cat:.4f}, dog_F1={ps_dog:.4f}")
        print(f"delta: acc={ps['delta_acc']:+.4f}, macro_F1={ps['delta_macro_f1']:+.4f}, cat_F1={ps_cat - sv_cat:+.4f}, dog_F1={ps_dog - sv_dog:+.4f}")
        print(f"pseudo_labeled_samples={ps['n_pseudo']}, accept_rate={ps['accept_rate']:.1%}, pseudo_label_accuracy={pa_str}")
        print()
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(2)
        ax.bar(x - 0.2, [sv_cat, sv_dog], 0.4,
               label="supervised",      color="steelblue", alpha=0.75, edgecolor="white")
        ax.bar(x + 0.2, [ps_cat, ps_dog], 0.4,
               label="pseudo-labeling", color="tomato",    alpha=0.75, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(["cat breeds (minority, 12)", "dog breeds (majority, 25)"])
        ax.set_ylabel("macro F1")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"PartA4 cat vs dog macro F1 at {int(frac*100)}% labeled data")
        ax.legend()
        for vals, offset in [([sv_cat, sv_dog], -0.2), ([ps_cat, ps_dog], 0.2)]:
            for j, v in enumerate(vals):
                ax.text(j + offset, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"PartA4_cat_dog_F1_{int(frac*100)}pct.png"), dpi=150)
        plt.close()

def main():
    out_dir = RESULTS_DIR_A
    os.makedirs(out_dir, exist_ok=True)
    ResetCSV(os.path.join(out_dir, "supervised_baseline.csv"))

    backbone  = "resnet34"
    best_l = 3
    n_epochs = 10
    n_epochs_per_stage = 3
    threshold = DEFAULT_THRESHOLD

    data = PrepareData()

    print(f"device={DEVICE}")
    print(f"backbone={backbone}, best_l={best_l}, threshold={threshold}")
    print(f"dataset=Oxford-IIIT Pet, n_classes=37, n_cat={len(data['cat_indices'])}, n_dog={len(data['dog_indices'])}")
    print(f"n_train={len(data['train_indices'])}, n_val={len(data['val_indices'])}, n_test={len(data['test_mc'])}")
    print(f"labeled_fractions={LABELED_FRACTIONS}, strategy_per_frac={STRATEGY_PER_FRAC}")
    print()

    print("Part A1: supervised baselines")
    res_sup = RunSupervisedBaselines(data, backbone, best_l, n_epochs, n_epochs_per_stage, out_dir)

    print(f"Part A2: pseudo-labeling, threshold={threshold}")
    res_pseudo = RunPseudoLabeling(data, backbone, best_l, n_epochs, n_epochs_per_stage, threshold, out_dir, supervised_results=res_sup)

    print(f"Part A3: threshold ablation, fraction={int(ABLATION_FRACTION*100)}%")
    res_ablation = RunThresholdAblation(data, backbone, best_l, n_epochs, n_epochs_per_stage, out_dir)

    print("Part A4: comparison and failure analysis")
    PlotComparison(res_sup, res_pseudo, out_dir)
    PlotThresholdAblation(res_ablation, out_dir)
    RunFailureAnalysis(data, res_sup, res_pseudo, out_dir)

    print("Summary:")
    print(f"backbone={backbone}, best_l={best_l}, threshold={threshold}")
    print()

    print("supervised baselines:")
    for frac in sorted(LABELED_FRACTIONS, reverse=True):
        r = res_sup[frac]
        print(f"fraction={int(frac*100)}%, strategy={r['strategy']}, n_labeled={len(r['labeled_idx'])}, test_accuracy={r['test_acc']:.4f}, macro_F1={r['macro_f1']:.4f}")

    print()
    print("pseudo-labeling vs supervised:")
    for frac in sorted(LABELED_FRACTIONS, reverse=True):
        sv = res_sup[frac]
        ps = res_pseudo.get(frac, {})
        st_acc = ps.get("student_test_acc", sv["test_acc"])
        n_ps = ps.get("n_pseudo", 0)
        ar = ps.get("accept_rate", 0.0)
        da = ps.get("delta_acc", 0.0)
        pa = ps.get("pseudo_acc")
        pa_str = f"{pa:.4f}" if pa is not None else "NA"
        print(f"fraction={int(frac*100)}%, supervised_test={sv['test_acc']:.4f}, pseudo_test={st_acc:.4f}, delta={da:+.4f}, n_pseudo={n_ps}, accept_rate={ar:.1%}, pseudo_label_acc={pa_str}")

    print()
    print("threshold ablation:")
    ab = res_ablation
    print(f"teacher: test_accuracy={ab['teacher']['test_acc']:.4f}, macro_F1={ab['teacher']['macro_f1']:.4f}, best_val_acc={ab['teacher']['best_val']:.4f}")
    for thr in sorted(THRESHOLD_VALUES):
        r = ab[thr]
        pa = r.get("pseudo_acc")
        pa_str = f"{pa:.4f}" if pa is not None else "NA"
        print(f"threshold={thr}, n_pseudo={r['n_pseudo']}, accept_rate={r['accept_rate']:.1%}, pseudo_label_acc={pa_str}, best_val_acc={r['best_val']:.4f}, test_accuracy={r['test_acc']:.4f}, delta={r['delta_acc']:+.4f}, macro_F1={r['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
