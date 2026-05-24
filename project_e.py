import os
import time
import copy
import csv
import random
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from torchvision import transforms, models
from torchvision.datasets import OxfordIIITPet
from sklearn.metrics import f1_score

# Setup
SEED = 42
# Use GPU when available; otherwise fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
RESULTS_DIR = "./results"
BATCH_SIZE = 32
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
VAL_FRACTION = 0.15
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 12 cat breeds in Oxford-IIIT Pet
CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay",
    "British Shorthair", "British_Shorthair",
    "Egyptian Mau", "Egyptian_Mau",
    "Maine Coon", "Maine_Coon",
    "Persian", "Ragdoll",
    "Russian Blue", "Russian_Blue",
    "Siamese", "Sphynx",
}

# ResNet18/34/50 share the same top-level layer names
RESNET_LAYER_GROUPS = ["layer4", "layer3", "layer2", "layer1", "conv1_bn1"]

SUMMARY_HEADER = ["experiment", "config", "best_val_acc", "test_acc", "train_time_sec"]

# Data augmentation
def GetTransforms(augment):
    # Create image preprocessing pipelines for training/validation
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if augment:
        train_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_tf = val_tf
    return train_tf, val_tf

# Cat/dog labelling and binary-task wrapper
def IsCat(class_name):
    # Return True if the class name belongs to one of the cat breeds
    return class_name in CAT_BREEDS

def GetLabels(dataset):
    # Extract labels from normal datasets, subsets, or custom datasets
    if isinstance(dataset, Subset):
        base = GetLabels(dataset.dataset)
        return [base[i] for i in dataset.indices]
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "_labels"):
        return list(dataset._labels)
    return [dataset[i][1] for i in range(len(dataset))]

class BinaryPetDataset(torch.utils.data.Dataset):
    # Wrap the 37-class pet dataset into a binary cat-vs-dog dataset
    def __init__(self, base):
        self.base = base
        labels = GetLabels(base)
        classes = base.classes
        self.targets = [0 if IsCat(classes[l]) else 1 for l in labels]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return img, self.targets[idx]

# Stratified train/val split and subset sampling
def StratifiedSplit(dataset, val_fraction, seed=SEED):
    # Split data into train/validation while keeping each class proportion similar
    labels = GetLabels(dataset)
    by_class = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_class[lbl].append(i)
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for lbl, idxs in by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_fraction))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx

def StratifiedFraction(indices, labels, fraction, seed=SEED):
    # Select a smaller stratified subset, used for limited-data experiments
    by_class = defaultdict(list)
    for idx, lbl in zip(indices, labels):
        by_class[lbl].append(idx)
    rng = random.Random(seed)
    sel = []
    for lbl, idxs in by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n = max(1, round(len(idxs) * fraction))
        sel.extend(idxs[:n])
    return sel

def ImbalancedSubset(indices, labels, minority_classes, minority_frac, seed=SEED):
    # Artificially reduce minority classes to create an imbalanced dataset
    by_class = defaultdict(list)
    for idx, lbl in zip(indices, labels):
        by_class[lbl].append(idx)
    rng = random.Random(seed)
    sel = []
    for lbl, idxs in by_class.items():
        idxs = idxs.copy()
        if lbl in minority_classes:
            rng.shuffle(idxs)
            n = max(1, round(len(idxs) * minority_frac))
            sel.extend(idxs[:n])
        else:
            sel.extend(idxs)
    return sel

# Download Oxford-IIIT Pet and produce all train/val/test splits
def PrepareData():
    # Download/load the dataset and prepare all train/val/test splits
    train_tf_aug, val_tf = GetTransforms(augment=True)
    trainval_aug = OxfordIIITPet(DATA_DIR, split="trainval", target_types="category",
                                   transform=train_tf_aug, download=True)
    trainval_clean = OxfordIIITPet(DATA_DIR, split="trainval", target_types="category",
                                     transform=val_tf, download=True)
    test_mc = OxfordIIITPet(DATA_DIR, split="test", target_types="category",
                              transform=val_tf, download=True)

    class_names = trainval_clean.classes
    cat_indices = {i for i, n in enumerate(class_names) if IsCat(n)}
    dog_indices = {i for i, n in enumerate(class_names) if not IsCat(n)}

    train_idx, val_idx = StratifiedSplit(trainval_clean, VAL_FRACTION)
    all_labels = GetLabels(trainval_clean)
    train_labels = [all_labels[i] for i in train_idx]

    train_mc = Subset(trainval_clean, train_idx)
    val_mc = Subset(trainval_clean, val_idx)
    trainval_clean_bin = BinaryPetDataset(trainval_clean)
    test_bin = BinaryPetDataset(test_mc)
    train_bin = Subset(trainval_clean_bin, train_idx)
    val_bin = Subset(trainval_clean_bin, val_idx)

    return {
        "trainval_aug": trainval_aug,
        "trainval_clean": trainval_clean,
        "test_mc": test_mc,
        "train_mc": train_mc,
        "val_mc": val_mc,
        "train_bin": train_bin,
        "val_bin": val_bin,
        "test_bin": test_bin,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "train_labels": train_labels,
        "class_names": class_names,
        "cat_indices": cat_indices,
        "dog_indices": dog_indices,
    }

# DataLoaders and class weights for imbalanced training
def MakeLoader(dataset, shuffle=True, sampler=None):
    # Convert a dataset into a PyTorch DataLoader for mini-batch training
    return DataLoader(dataset, batch_size=BATCH_SIZE,
                      shuffle=(shuffle and sampler is None),
                      sampler=sampler, num_workers=NUM_WORKERS,
                      pin_memory=DEVICE.type == "cuda")

def MakeWeightedLoader(dataset):
    # Build a DataLoader that oversamples rare classes using inverse-frequency weights
    labels = GetLabels(dataset)
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return MakeLoader(dataset, shuffle=False, sampler=sampler)

def ComputeClassWeights(dataset, num_classes):
    # Compute class weights for weighted cross-entropy loss
    labels = GetLabels(dataset)
    counts = Counter(labels)
    total = sum(counts.values())
    w = torch.zeros(num_classes)
    for c, n in counts.items():
        w[c] = total / (num_classes * n)
    return w

# Build pretrained ResNet and configure which layers are trainable
def BuildBackbone(backbone, num_classes, pretrained=True):
    # Load a pretrained ResNet and replace its final classifier layer
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=weights)
    elif backbone == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet34(weights=weights)
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet50(weights=weights)
    else:
        raise ValueError(backbone)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def SetRequiresGrad(module, value):
    for p in module.parameters():
        p.requires_grad = value

def FreezeAllExceptFC(model):
    # Freeze the CNN feature extractor and train only the final fully connected layer
    SetRequiresGrad(model, False)
    SetRequiresGrad(model.fc, True)

def UnfreezeLastL(model, l):
    # Unfreeze the last l ResNet layer groups for fine-tuning
    FreezeAllExceptFC(model)
    for grp in RESNET_LAYER_GROUPS[:l]:
        if grp == "conv1_bn1":
            SetRequiresGrad(model.conv1, True)
            SetRequiresGrad(model.bn1, True)
        else:
            SetRequiresGrad(getattr(model, grp), True)

def CountTrainable(model):
    # Count how many parameters will actually be updated during training
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Single-epoch training and evaluation
def TrainOneEpoch(model, loader, criterion, optimizer):
    # Run one full pass over the training data and update model weights
    model.train()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

def Evaluate(model, loader, criterion):
    # Evaluate model performance without updating weights
    model.eval()
    loss_sum = correct = total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            preds = out.argmax(1)
            correct += preds.eq(y).sum().item()
            total += x.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    return loss_sum / total, correct / total, all_preds, all_labels

def EvaluateTTA(model, loader, criterion):
    # Test-time augmentation.
    model.eval()
    loss_sum = correct = total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits1 = model(x)
            logits2 = model(torch.flip(x, dims=[3]))
            logits = (logits1 + logits2) / 2
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += preds.eq(y).sum().item()
            total += x.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    return loss_sum / total, correct / total, all_preds, all_labels

# Optimizer factory and full training loop (best-by-val selection)
def MakeOptimizer(model, lr, wd, opt_type):
    # Create the selected optimizer using only trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_type == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    if opt_type == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if opt_type == "nag":
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    raise ValueError(opt_type)

def TrainModel(model, train_loader, val_loader, n_epochs, lr,
    # Full training loop with validation tracking and best-model checkpointing
               wd=0.0, opt_type="adam", class_weights=None, label_smoothing=0.0):
    cw = class_weights.to(DEVICE) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    optimizer = MakeOptimizer(model, lr, wd, opt_type)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = 0.0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(n_epochs):
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

    model.load_state_dict(best_wts)
    return model, history, best_val

# Plotting helpers and CSV writers for saving results
def PlotHistory(history, title, save_path=None):
    # Plot loss and accuracy curves for training and validation
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="training loss")
    ax1.plot(epochs, history["val_loss"], label="validation loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title(f"{title} - loss")
    ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="training acc")
    ax2.plot(epochs, history["val_acc"], label="validation acc")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title(f"{title} - accuracy")
    ax2.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.close()

def BarChart(names, values, title, ylabel="test accuracy", save_path=None):
    # Save a simple bar chart comparing experiment results
    plt.figure(figsize=(max(6, len(names) * 0.9), 4))
    bars = plt.bar(names, values, color="steelblue", alpha=0.75, edgecolor="white")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, min(1.05, max(values) + 0.1))
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.close()

def PlotPerClassF1(labels_true, labels_pred, class_names, title,
    # Plot F1 score for every class, optionally highlighting minority classes
                    highlight_indices=None, save_path=None):
    f1s = f1_score(labels_true, labels_pred, average=None,
                    labels=list(range(len(class_names))), zero_division=0)
    colors = ["tomato" if (highlight_indices and i in highlight_indices)
              else "steelblue" for i in range(len(class_names))]
    plt.figure(figsize=(16, 4))
    plt.bar(class_names, f1s, color=colors, edgecolor="white")
    plt.ylabel("F1 score")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    if highlight_indices:
        plt.legend(handles=[
            mpatches.Patch(color="tomato", label="minority (cat breeds)"),
            mpatches.Patch(color="steelblue", label="majority (dog breeds)"),
        ])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.close()

def AppendCSV(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)

def ResetCSV(path):
    if os.path.exists(path):
        os.remove(path)

# Part 1: binary cat-vs-dog sanity check (target test acc >= 99%)
def RunBinary(data, backbone, n_epochs, out_dir):
    # Part 1: Train a binary cat-vs-dog classifier as a sanity check
    train_loader = MakeLoader(data["train_bin"])
    val_loader = MakeLoader(data["val_bin"], shuffle=False)
    test_loader = MakeLoader(data["test_bin"], shuffle=False)

    model = BuildBackbone(backbone, 2, pretrained=True)
    FreezeAllExceptFC(model)   # only train the replaced fc layer, all conv layers frozen
    model = model.to(DEVICE)

    t_start = time.time()
    model, history, best_val = TrainModel(
        model, train_loader, val_loader,
        n_epochs=n_epochs, lr=1e-3, opt_type="adam",
    )
    train_time = time.time() - t_start

    _, test_acc, preds, lbls = Evaluate(model, test_loader, nn.CrossEntropyLoss())
    macro_f1 = f1_score(lbls, preds, average="macro")
    weighted_f1 = f1_score(lbls, preds, average="weighted")

    tag = f"backbone={backbone}, mode=fc-only, optimizer=Adam, eta=1e-3, n_epochs={n_epochs}"
    print(tag)
    print(f"best_val_acc={best_val:.4f}")
    print(f"test_accuracy={test_acc:.4f}")
    print(f"macro_F1={macro_f1:.4f}")
    print(f"weighted_F1={weighted_f1:.4f}")
    print(f"train_time={train_time:.1f}s")
    print()

    PlotHistory(history, "Part1 Binary Classification (fc-only)",
                save_path=os.path.join(out_dir, "Part1_binary.png"))
    AppendCSV(os.path.join(out_dir, "summary.csv"), SUMMARY_HEADER,
        ["Part1_Binary", tag, best_val, test_acc, round(train_time, 1)])
    return {"test_acc": test_acc, "best_val": best_val, "history": history,
            "train_time": train_time, "macro_f1": macro_f1, "weighted_f1": weighted_f1}

# Part 2: 37-class linear probing on frozen ImageNet features
def RunLinearProbe(data, backbone, n_epochs, out_dir):
    # Part 2: Train only the final layer for 37-class breed classification
    train_loader = MakeLoader(data["train_mc"])
    val_loader = MakeLoader(data["val_mc"], shuffle=False)
    test_loader = MakeLoader(data["test_mc"], shuffle=False)

    model = BuildBackbone(backbone, 37, pretrained=True)
    FreezeAllExceptFC(model)
    model = model.to(DEVICE)

    t_start = time.time()
    model, history, best_val = TrainModel(
        model, train_loader, val_loader,
        n_epochs=n_epochs, lr=1e-3, opt_type="adam",
    )
    train_time = time.time() - t_start

    _, test_acc, preds, lbls = Evaluate(model, test_loader, nn.CrossEntropyLoss())
    macro_f1 = f1_score(lbls, preds, average="macro")
    weighted_f1 = f1_score(lbls, preds, average="weighted")

    tag = f"backbone={backbone}, mode=fc-only, optimizer=Adam, eta=1e-3, n_epochs={n_epochs}"
    print(tag)
    print(f"best_val_acc={best_val:.4f}")
    print(f"test_accuracy={test_acc:.4f}")
    print(f"macro_F1={macro_f1:.4f}")
    print(f"weighted_F1={weighted_f1:.4f}")
    print(f"train_time={train_time:.1f}s")
    print()

    PlotHistory(history, "Part2 Linear Probing (fc-only, frozen conv)",
                save_path=os.path.join(out_dir, "Part2_linear_probing.png"))
    AppendCSV(os.path.join(out_dir, "summary.csv"), SUMMARY_HEADER,
        ["Part2_LinearProbe", tag, best_val, test_acc, round(train_time, 1)])
    return {"test_acc": test_acc, "best_val": best_val, "history": history,
            "train_time": train_time}

# Part 3: Strategy 1: fine-tune the last l layer groups simultaneously, sweep l
def RunStrategy1(data, backbone, max_l, n_epochs, out_dir):
    # Part 3: Fine-tune the last l ResNet layer groups and compare l values
    train_loader = MakeLoader(data["train_mc"])
    val_loader = MakeLoader(data["val_mc"], shuffle=False)
    test_loader = MakeLoader(data["test_mc"], shuffle=False)

    results = {}
    for l in range(1, max_l + 1):
        model = BuildBackbone(backbone, 37, pretrained=True)
        UnfreezeLastL(model, l)
        model = model.to(DEVICE)
        lr = 1e-4 if l > 1 else 5e-4
        n_params = CountTrainable(model)

        t_start = time.time()
        model, history, best_val = TrainModel(
            model, train_loader, val_loader,
            n_epochs=n_epochs, lr=lr,
            opt_type="adamw", wd=1e-3, label_smoothing=0.1,
        )
        train_time = time.time() - t_start
        _, test_acc, _, _ = EvaluateTTA(model, test_loader, nn.CrossEntropyLoss())

        results[l] = {"test_acc": test_acc, "best_val": best_val,
                      "history": history, "train_time": train_time,
                      "n_params": n_params, "lr": lr}

        tag = f"backbone={backbone}, l={l}, optimizer=AdamW, eta={lr}, wd=1e-3, LS=0.1, n_epochs={n_epochs}, TTA=yes"
        print(tag)
        print(f"trainable_params={n_params}")
        print(f"best_val_acc={best_val:.4f}")
        print(f"test_accuracy={test_acc:.4f}")
        print(f"train_time={train_time:.1f}s")
        print()

        PlotHistory(history, f"Part3 Strategy1 l={l}",
                    save_path=os.path.join(out_dir, f"Part3_strategy1_l{l}.png"))
        AppendCSV(os.path.join(out_dir, "summary.csv"), SUMMARY_HEADER,
            [f"Part3_Strategy1_l{l}", tag, best_val, test_acc, round(train_time, 1)])

    best_l = max(results, key=lambda l: results[l]["best_val"])
    print(f"best_l_by_val_acc={best_l}, val_acc={results[best_l]['best_val']:.4f}, "
          f"test_accuracy={results[best_l]['test_acc']:.4f}")
    print()

    ls = sorted(results.keys())
    BarChart([f"l={l}" for l in ls],
             [results[l]["test_acc"] for l in ls],
             "Part3 strategy1 summary",
             save_path=os.path.join(out_dir, "Part3_strategy1_summary.png"))
    return {"results": results, "best_l": best_l}

# Part 4: Strategy 2: gradual unfreezing (fc -> layer4 -> ... -> conv1)
def RunStrategy2(data, backbone, n_epochs_per_stage, out_dir):
    # Part 4: Gradually unfreeze deeper layers stage by stage
    train_loader = MakeLoader(data["train_mc"])
    val_loader = MakeLoader(data["val_mc"], shuffle=False)
    test_loader = MakeLoader(data["test_mc"], shuffle=False)

    model = BuildBackbone(backbone, 37, pretrained=True)
    FreezeAllExceptFC(model)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=5e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs_per_stage)

    best_val = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    t_start = time.time()

    for stage in range(len(RESNET_LAYER_GROUPS) + 1):
        for ep in range(n_epochs_per_stage):
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
                SetRequiresGrad(model.bn1, True)
            else:
                SetRequiresGrad(getattr(model, g), True)
            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-4, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs_per_stage)
    train_time = time.time() - t_start

    model.load_state_dict(best_wts)
    _, test_acc, _, _ = EvaluateTTA(model, test_loader, nn.CrossEntropyLoss())

    tag = (f"backbone={backbone}, mode=gradual_unfreezing, optimizer=AdamW, "
           f"wd=1e-3, LS=0.1, n_epochs_per_stage={n_epochs_per_stage}, TTA=yes")
    print(tag)
    print("stages=fc, +layer4, +layer3, +layer2, +layer1, +conv1+bn1")
    print(f"best_val_acc={best_val:.4f}")
    print(f"test_accuracy={test_acc:.4f}")
    print(f"train_time={train_time:.1f}s")
    print()

    PlotHistory(history, "Part4 Strategy2 Gradual Unfreezing",
                save_path=os.path.join(out_dir, "Part4_strategy2_gradual_unfreezing.png"))
    AppendCSV(os.path.join(out_dir, "summary.csv"), SUMMARY_HEADER,
        ["Part4_Strategy2_Gradual", tag, best_val, test_acc, round(train_time, 1)])
    return {"test_acc": test_acc, "best_val": best_val,
            "history": history, "train_time": train_time}


# Part 5: limited training data: sweep fraction x augmentation x L2
def RunLimitedData(data, backbone, best_l, n_epochs, out_dir):
    # Part 5: Test performance with less labelled training data, augmentation, and L2 regularization
    val_loader = MakeLoader(data["val_mc"], shuffle=False)
    test_loader = MakeLoader(data["test_mc"], shuffle=False)

    ResetCSV(os.path.join(out_dir, "limited_data_summary.csv"))
    configs = []
    for frac in [1.0, 0.1, 0.01]:
        for aug in [False, True]:
            for wd in [0.0, 1e-4, 1e-3]:
                configs.append({"fraction": frac, "augment": aug, "weight_decay": wd})

    results = {}
    for cfg in configs:
        frac = cfg["fraction"]
        aug = cfg["augment"]
        wd = cfg["weight_decay"]
        sub_idx = StratifiedFraction(data["train_indices"], data["train_labels"], frac)
        base_ds = data["trainval_aug"] if aug else data["trainval_clean"]
        train_subset = Subset(base_ds, sub_idx)
        train_loader = MakeLoader(train_subset)

        model = BuildBackbone(backbone, 37, pretrained=True)
        UnfreezeLastL(model, best_l)
        model = model.to(DEVICE)

        t_start = time.time()
        model, history, best_val = TrainModel(
            model, train_loader, val_loader,
            n_epochs=n_epochs, lr=1e-4, wd=wd, opt_type="adam",
        )
        train_time = time.time() - t_start
        _, test_acc, _, _ = Evaluate(model, test_loader, nn.CrossEntropyLoss())

        key = f"frac={frac}_aug={aug}_wd={wd}"
        results[key] = {"test_acc": test_acc, "best_val": best_val,
                        "history": history, "train_time": train_time,
                        "n_train": len(sub_idx), "cfg": cfg}

        tag = f"fraction={frac}, augment={aug}, weight_decay={wd}, n_train={len(sub_idx)}"
        print(tag)
        print(f"best_val_acc={best_val:.4f}")
        print(f"test_accuracy={test_acc:.4f}")
        print(f"train_time={train_time:.1f}s")
        print()

        AppendCSV(os.path.join(out_dir, "limited_data_summary.csv"),
            ["fraction", "augment", "weight_decay", "n_train",
             "best_val_acc", "test_acc", "train_time_sec"],
            [frac, aug, wd, len(sub_idx), best_val, test_acc, round(train_time, 1)])
        AppendCSV(os.path.join(out_dir, "summary.csv"), SUMMARY_HEADER,
            [f"Part5_frac{frac}_aug{aug}_wd{wd}",
             f"{backbone}, l={best_l}, " + tag,
             best_val, test_acc, round(train_time, 1)])

    print("Limited-data summary (best config selected by val_acc):")
    for frac in [1.0, 0.1, 0.01]:
        best_key = max((k for k in results if f"frac={frac}_" in k),
                       key=lambda k: results[k]["best_val"])   # use val, not test
        r = results[best_key]
        cfg = r["cfg"]
        print(f"fraction={frac}: best augment={cfg['augment']}, "
              f"weight_decay={cfg['weight_decay']}, "
              f"val_acc={r['best_val']:.4f}, test_accuracy={r['test_acc']:.4f}")
    print()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, frac in zip(axes, [1.0, 0.1, 0.01]):
        for aug in [False, True]:
            for wd in [0.0, 1e-4, 1e-3]:
                key = f"frac={frac}_aug={aug}_wd={wd}"
                ax.plot(results[key]["history"]["val_acc"],
                        label=f"aug={aug}, wd={wd}")
        ax.set_title(f"fraction = {int(frac*100)}%")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation accuracy")
        ax.legend(fontsize=7)
    plt.suptitle("Part 5 limited data")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Part5_limited_data.png"), dpi=200)
    plt.close()
    return results

# Part 6: imbalanced classes (cat=20%): no comp / weighted CE / oversampling
def RunImbalanced(data, backbone, best_l, n_epochs, out_dir):
    # Part 6: Compare ways to handle class imbalance
    val_loader = MakeLoader(data["val_mc"], shuffle=False)
    test_loader = MakeLoader(data["test_mc"], shuffle=False)

    imb_idx = ImbalancedSubset(data["train_indices"], data["train_labels"],
                                 data["cat_indices"], 0.20)
    train_imb = Subset(data["trainval_clean"], imb_idx)

    ResetCSV(os.path.join(out_dir, "imbalance_summary.csv"))

    def RunOne(strategy_name, train_loader, class_weights=None):
        model = BuildBackbone(backbone, 37, pretrained=True)
        UnfreezeLastL(model, best_l)
        model = model.to(DEVICE)
        t_start = time.time()
        model, history, best_val = TrainModel(
            model, train_loader, val_loader,
            n_epochs=n_epochs, lr=1e-4, opt_type="adam",
            class_weights=class_weights,
        )
        train_time = time.time() - t_start
        _, test_acc, preds, lbls = Evaluate(model, test_loader, nn.CrossEntropyLoss())
        return {"test_acc": test_acc, "best_val": best_val,
                "history": history, "preds": preds, "labels": lbls,
                "train_time": train_time, "strategy": strategy_name}

    print(f"minority=12 cat breeds reduced to 20% of training samples, "
          f"majority=25 dog breeds unchanged, "
          f"imbalanced_train_size={len(imb_idx)}, original_train_size={len(data['train_indices'])}")
    print()

    res_A = RunOne("no_compensation", MakeLoader(train_imb))
    cw = ComputeClassWeights(train_imb, 37)
    res_B = RunOne("weighted_CE", MakeLoader(train_imb), class_weights=cw)
    res_C = RunOne("oversampling", MakeWeightedLoader(train_imb))

    for tag, res in [("no_compensation", res_A),
                     ("weighted_CE", res_B),
                     ("oversampling", res_C)]:
        PlotPerClassF1(res["labels"], res["preds"], data["class_names"],
                        f"Part6 {tag} per-class F1",
                        highlight_indices=data["cat_indices"],
                        save_path=os.path.join(out_dir, f"Part6_{tag}_per_class_F1.png"))

    metrics = {}
    for tag, res in [("no_compensation", res_A),
                     ("weighted_CE", res_B),
                     ("oversampling", res_C)]:
        preds, lbls = res["preds"], res["labels"]
        cat_lbls = [l for l in lbls if l in data["cat_indices"]]
        dog_lbls = [l for l in lbls if l in data["dog_indices"]]
        cat_preds = [p for p, l in zip(preds, lbls) if l in data["cat_indices"]]
        dog_preds = [p for p, l in zip(preds, lbls) if l in data["dog_indices"]]
        cat_acc = sum(p == l for p, l in zip(cat_preds, cat_lbls)) / max(len(cat_lbls), 1)
        dog_acc = sum(p == l for p, l in zip(dog_preds, dog_lbls)) / max(len(dog_lbls), 1)
        macro_f1 = f1_score(lbls, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(lbls, preds, average="weighted", zero_division=0)
        cat_f1 = f1_score(cat_lbls, cat_preds,
                          labels=sorted(list(data["cat_indices"])),
                          average="macro", zero_division=0)
        dog_f1 = f1_score(dog_lbls, dog_preds,
                          labels=sorted(list(data["dog_indices"])),
                          average="macro", zero_division=0)

        metrics[tag] = {"test_acc": res["test_acc"], "cat_acc": cat_acc, "dog_acc": dog_acc,
                        "cat_f1": cat_f1, "dog_f1": dog_f1,
                        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
                        "train_time": res["train_time"], "best_val": res["best_val"]}

        cfg_tag = f"backbone={backbone}, strategy={tag}, l={best_l}, n_epochs={n_epochs}"
        print(cfg_tag)
        print(f"test_accuracy={res['test_acc']:.4f}")
        print(f"cat_acc={cat_acc:.4f}, dog_acc={dog_acc:.4f}")
        print(f"cat_F1={cat_f1:.4f}, dog_F1={dog_f1:.4f}")
        print(f"macro_F1={macro_f1:.4f}, weighted_F1={weighted_f1:.4f}")
        print(f"train_time={res['train_time']:.1f}s")
        print()

        AppendCSV(os.path.join(out_dir, "imbalance_summary.csv"),
            ["strategy", "best_val_acc", "test_acc", "cat_acc", "dog_acc",
             "cat_f1", "dog_f1", "macro_f1", "weighted_f1", "train_time_sec"],
            [tag, res["best_val"], res["test_acc"], cat_acc, dog_acc,
             cat_f1, dog_f1, macro_f1, weighted_f1, round(res["train_time"], 1)])
        AppendCSV(os.path.join(out_dir, "summary.csv"), SUMMARY_HEADER,
            [f"Part6_{tag}", cfg_tag, res["best_val"], res["test_acc"],
             round(res["train_time"], 1)])

    print("Imbalance strategy comparison on test set:")
    for t in ["no_compensation", "weighted_CE", "oversampling"]:
        print(f"  {t}: test_accuracy={metrics[t]['test_acc']:.4f}, "
              f"macro_F1={metrics[t]['macro_f1']:.4f}, "
              f"weighted_F1={metrics[t]['weighted_f1']:.4f}")
    print()

    BarChart(["no_compensation", "weighted_CE", "oversampling"],
             [res_A["test_acc"], res_B["test_acc"], res_C["test_acc"]],
             "Part6 imbalanced summary",
             save_path=os.path.join(out_dir, "Part6_imbalanced_summary.png"))
    return {"A": res_A, "B": res_B, "C": res_C, "metrics": metrics}

# Part 5b: Strategy 1 vs Strategy 2 under limited data
def RunLimitedDataStrategyComparison(data, backbone, best_l, n_epochs,
    # Part 5b: Compare Strategy 1 and Strategy 2 when labelled data becomes scarce
                                      n_epochs_per_stage, out_dir):
    val_loader  = MakeLoader(data["val_mc"], shuffle=False)
    test_loader = MakeLoader(data["test_mc"], shuffle=False)

    fractions = [1.0, 0.1, 0.01]
    results = {}
    ResetCSV(os.path.join(out_dir, "limited_strategy_comparison.csv"))

    for frac in fractions:
        sub_idx = StratifiedFraction(data["train_indices"], data["train_labels"], frac)
        train_subset = Subset(data["trainval_aug"], sub_idx)   # augmented
        train_loader = MakeLoader(train_subset)
        n_train = len(sub_idx)

        # Strategy 1: unfreeze last best_l groups simultaneously
        model_s1 = BuildBackbone(backbone, 37, pretrained=True)
        UnfreezeLastL(model_s1, best_l)
        model_s1 = model_s1.to(DEVICE)
        t0 = time.time()
        model_s1, _, best_val_s1 = TrainModel(
            model_s1, train_loader, val_loader,
            n_epochs=n_epochs, lr=1e-4, wd=1e-4, opt_type="adamw",
            label_smoothing=0.1,
        )
        time_s1 = time.time() - t0
        _, test_acc_s1, _, _ = EvaluateTTA(model_s1, test_loader, nn.CrossEntropyLoss())

        # Strategy 2: gradual unfreezing fc → layer4 → … → conv1 ---
        model_s2 = BuildBackbone(backbone, 37, pretrained=True)
        FreezeAllExceptFC(model_s2)
        model_s2 = model_s2.to(DEVICE)
        criterion_s2 = nn.CrossEntropyLoss(label_smoothing=0.1)
        opt_s2 = optim.AdamW([p for p in model_s2.parameters() if p.requires_grad],
                              lr=5e-4, weight_decay=1e-3)
        sch_s2 = optim.lr_scheduler.CosineAnnealingLR(opt_s2, T_max=n_epochs_per_stage)
        best_val_s2 = 0.0
        best_wts_s2 = copy.deepcopy(model_s2.state_dict())
        t0 = time.time()
        for stage in range(len(RESNET_LAYER_GROUPS) + 1):
            for _ in range(n_epochs_per_stage):
                TrainOneEpoch(model_s2, train_loader, criterion_s2, opt_s2)
                _, vl_acc, _, _ = Evaluate(model_s2, val_loader, criterion_s2)
                sch_s2.step()
                if vl_acc > best_val_s2:
                    best_val_s2 = vl_acc
                    best_wts_s2 = copy.deepcopy(model_s2.state_dict())
            if stage < len(RESNET_LAYER_GROUPS):
                g = RESNET_LAYER_GROUPS[stage]
                if g == "conv1_bn1":
                    SetRequiresGrad(model_s2.conv1, True)
                    SetRequiresGrad(model_s2.bn1, True)
                else:
                    SetRequiresGrad(getattr(model_s2, g), True)
                opt_s2 = optim.AdamW(
                    [p for p in model_s2.parameters() if p.requires_grad],
                    lr=1e-4, weight_decay=1e-3)
                sch_s2 = optim.lr_scheduler.CosineAnnealingLR(
                    opt_s2, T_max=n_epochs_per_stage)
        time_s2 = time.time() - t0
        model_s2.load_state_dict(best_wts_s2)
        _, test_acc_s2, _, _ = EvaluateTTA(model_s2, test_loader, nn.CrossEntropyLoss())

        results[frac] = {
            "s1": {"test_acc": test_acc_s1, "best_val": best_val_s1, "train_time": time_s1},
            "s2": {"test_acc": test_acc_s2, "best_val": best_val_s2, "train_time": time_s2},
            "n_train": n_train,
        }

        print(f"fraction={frac}, n_train={n_train}")
        print(f"  strategy1_l{best_l}: best_val={best_val_s1:.4f}, "
              f"test_acc={test_acc_s1:.4f}, train_time={time_s1:.1f}s")
        print(f"  strategy2_gradual:   best_val={best_val_s2:.4f}, "
              f"test_acc={test_acc_s2:.4f}, train_time={time_s2:.1f}s")
        print()

        for strat, ta, bv, tt in [
            (f"strategy1_l{best_l}", test_acc_s1, best_val_s1, time_s1),
            ("strategy2_gradual",    test_acc_s2, best_val_s2, time_s2),
        ]:
            AppendCSV(os.path.join(out_dir, "limited_strategy_comparison.csv"),
                ["fraction", "n_train", "strategy", "best_val_acc", "test_acc", "train_time_sec"],
                [frac, n_train, strat, bv, ta, round(tt, 1)])
            AppendCSV(os.path.join(out_dir, "summary.csv"), SUMMARY_HEADER,
                [f"Part5b_{strat}_frac{frac}",
                 f"backbone={backbone}, {strat}, frac={frac}, aug=True, wd=1e-4",
                 bv, ta, round(tt, 1)])

    # Side-by-side bar chart
    plt.figure(figsize=(8, 5))
    fr = sorted(results.keys(), reverse=True)
    s1_accs = [results[f]["s1"]["test_acc"] for f in fr]
    s2_accs = [results[f]["s2"]["test_acc"] for f in fr]
    x = np.arange(len(fr))
    plt.bar(x - 0.2, s1_accs, 0.4, label=f"Strategy 1 (l={best_l})", color="steelblue")
    plt.bar(x + 0.2, s2_accs, 0.4, label="Strategy 2 (gradual)", color="tomato")
    plt.xticks(x, [f"{int(f * 100)}%" for f in fr])
    plt.xlabel("training fraction")
    plt.ylabel("test accuracy")
    plt.title("Part 5b: Strategy 1 vs Strategy 2 under limited data")
    plt.ylim(0, 1.05)
    for i, (v1, v2) in enumerate(zip(s1_accs, s2_accs)):
        plt.text(i - 0.2, v1 + 0.01, f"{v1:.3f}", ha="center", fontsize=8)
        plt.text(i + 0.2, v2 + 0.01, f"{v2:.3f}", ha="center", fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Part5b_strategy_comparison_limited_data.png"), dpi=200)
    plt.close()

    print("Strategy comparison under limited data summary:")
    for frac in sorted(results.keys(), reverse=True):
        r = results[frac]
        winner = "S1" if r["s1"]["best_val"] >= r["s2"]["best_val"] else "S2"
        print(f"  fraction={frac}: "
              f"S1 val={r['s1']['best_val']:.4f}, test={r['s1']['test_acc']:.4f}; "
              f"S2 val={r['s2']['best_val']:.4f}, test={r['s2']['test_acc']:.4f} "
              f"-> {winner} selected by val")
    print()
    return results


def main():
    # Main entry point: runs all experiment parts in order
    out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    ResetCSV(os.path.join(out_dir, "summary.csv"))

    backbone = "resnet34"
    n_epochs_binary = 10
    n_epochs_multiclass = 10
    n_epochs_per_stage = 3
    max_l = 5

    data = PrepareData()

    print(f"device={DEVICE}")
    print(f"backbone={backbone}")
    print(f"dataset=Oxford-IIIT Pet")
    print(f"n_classes=37, n_cat_breeds={len(data['cat_indices'])}, "
          f"n_dog_breeds={len(data['dog_indices'])}")
    print(f"n_train={len(data['train_indices'])}, "
          f"n_val={len(data['val_indices'])}, "
          f"n_test={len(data['test_mc'])}")
    print("note: test set is only used for final evaluation")
    print()

    print("Part 1: binary classification")
    res_binary = RunBinary(data, backbone, n_epochs_binary, out_dir)

    print("Part 2: 37-class linear probing")
    res_lp = RunLinearProbe(data, backbone, n_epochs_multiclass, out_dir)

    print("Part 3: strategy 1 (fine-tune last l layer groups)")
    res_s1 = RunStrategy1(data, backbone, max_l, n_epochs_multiclass, out_dir)
    best_l = res_s1["best_l"]

    print("Part 4: strategy 2 (gradual unfreezing)")
    res_s2 = RunStrategy2(data, backbone, n_epochs_per_stage, out_dir)

    print("Strategy comparison:")
    s1_best = res_s1["results"][best_l]
    print(f"strategy_1_best_l={best_l}, test_accuracy={s1_best['test_acc']:.4f}, "
          f"train_time={s1_best['train_time']:.1f}s")
    print(f"strategy_2_gradual, test_accuracy={res_s2['test_acc']:.4f}, "
          f"train_time={res_s2['train_time']:.1f}s")
    print()

    print(f"Part 5: limited training data (using l={best_l} from Part 3)")
    res_limited = RunLimitedData(data, backbone, best_l, n_epochs_multiclass, out_dir)

    print(f"Part 5b: Strategy 1 vs Strategy 2 under limited data (fractions=100%/10%/1%)")
    res_s_cmp = RunLimitedDataStrategyComparison(
        data, backbone, best_l,
        n_epochs=n_epochs_multiclass,
        n_epochs_per_stage=n_epochs_per_stage,
        out_dir=out_dir)

    print(f"Part 6: imbalanced classes (using l={best_l} from Part 3)")
    res_imb = RunImbalanced(data, backbone, best_l, n_epochs_multiclass, out_dir)

    print("Summary:")
    print(f"backbone={backbone}")
    print(f"binary, test_accuracy={res_binary['test_acc']:.4f}")
    print(f"linear_probing, test_accuracy={res_lp['test_acc']:.4f}")
    print(f"strategy_1_best_l={best_l}, test_accuracy={res_s1['results'][best_l]['test_acc']:.4f}")
    print(f"strategy_2_gradual, test_accuracy={res_s2['test_acc']:.4f}")
    for frac in [1.0, 0.1, 0.01]:
        best_key = max((k for k in res_limited if f"frac={frac}_" in k),
                       key=lambda k: res_limited[k]["best_val"])   # select by val
        print(f"limited fraction={frac}, best val_acc={res_limited[best_key]['best_val']:.4f}, "
              f"test_accuracy={res_limited[best_key]['test_acc']:.4f}")
    for frac in [1.0, 0.1, 0.01]:
        r = res_s_cmp[frac]
        print(f"limited_strategy_cmp fraction={frac}: "
              f"S1={r['s1']['test_acc']:.4f}, S2={r['s2']['test_acc']:.4f}")
    print(f"imbalance no_compensation, test_accuracy={res_imb['A']['test_acc']:.4f}")
    print(f"imbalance weighted_CE, test_accuracy={res_imb['B']['test_acc']:.4f}")
    print(f"imbalance oversampling, test_accuracy={res_imb['C']['test_acc']:.4f}")


if __name__ == "__main__":
    # Only run main() when this file is executed directly, not when imported
    main()
