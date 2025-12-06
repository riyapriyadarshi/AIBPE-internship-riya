from pathlib import Path
import argparse
import pandas as pd
from PIL import Image
import json
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import torch.optim as optim

class FairFaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transforms=None, max_samples=None):
        self.df = pd.read_csv(csv_file)
        if max_samples:
            self.df = self.df.sample(n=min(max_samples, len(self.df)), random_state=42).reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transforms = transforms

        cols = list(self.df.columns)
        self.fname_col = next(c for c in cols if "file" in c.lower() or "img" in c.lower())
        self.gender_col = next(c for c in cols if "gender" in c.lower())
        self.race_col = next(c for c in cols if "race" in c.lower())
        self.age_col = next(c for c in cols if "age" in c.lower())

        self.gender_labels = sorted(self.df[self.gender_col].unique().tolist())
        self.race_labels = sorted(self.df[self.race_col].unique().tolist())

        age_vals = self.df[self.age_col].unique()
        if all(isinstance(a, str) for a in age_vals):
            self.age_labels = sorted(list(set(age_vals)))
        else:
            self.age_labels = ["0-2","3-9","10-19","20-29","30-39","40-49","50-59","60+"]

    def __len__(self):
        return len(self.df)

    def age_to_bucket(self, a):
        try: a = int(a)
        except: return a
        if a <= 2: return "0-2"
        if a <= 9: return "3-9"
        if a <= 19: return "10-19"
        if a <= 29: return "20-29"
        if a <= 39: return "30-39"
        if a <= 49: return "40-49"
        if a <= 59: return "50-59"
        return "60+"

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row[self.fname_col]
        p = self.img_dir / fname
        if not p.exists():
            p = self.img_dir / Path(fname).name
        if not p.exists():
            found = list(self.img_dir.rglob(Path(fname).name))
            if not found:
                raise FileNotFoundError(f"Image not found: {fname}")
            p = found[0]

        img = Image.open(p).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        gender = row[self.gender_col]
        race = row[self.race_col]
        age_raw = row[self.age_col]
        age_bucket = age_raw if isinstance(age_raw, str) and "-" in age_raw else self.age_to_bucket(age_raw)

        g_idx = self.gender_labels.index(gender)
        r_idx = self.race_labels.index(race)
        a_idx = self.age_labels.index(age_bucket)
        return img, g_idx, a_idx, r_idx

class MultiHeadResNet(nn.Module):
    def __init__(self, n_gender, n_age, n_race, pretrained=True):
        super().__init__()
        base = models.resnet18(pretrained=pretrained)
        in_feat = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.gender_head = nn.Linear(in_feat, n_gender)
        self.age_head = nn.Linear(in_feat, n_age)
        self.race_head = nn.Linear(in_feat, n_race)

    def forward(self, x):
        feat = self.base(x)
        g = self.gender_head(feat)
        a = self.age_head(feat)
        r = self.race_head(feat)
        return g, a, r

def train_epoch(model, loader, opt, device, crit):
    model.train()
    total_loss = 0
    for imgs, g, a, r in tqdm(loader, desc="train"):
        imgs, g, a, r = imgs.to(device), g.to(device), a.to(device), r.to(device)
        opt.zero_grad()
        out_g, out_a, out_r = model(imgs)
        loss = crit(out_g, g) + crit(out_a, a) + crit(out_r, r)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, device, crit):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, g, a, r in tqdm(loader, desc="eval"):
            imgs, g, a, r = imgs.to(device), g.to(device), a.to(device), r.to(device)
            out_g, out_a, out_r = model(imgs)
            loss = crit(out_g, g) + crit(out_a, a) + crit(out_r, r)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/fairface")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--max_val_samples", type=int, default=400)
    parser.add_argument("--output", type=str, default="models/fairface/fairface_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "fairface_label_train.csv"
    val_csv = data_dir / "fairface_label_val.csv"
    train_img_dir = data_dir / "train"
    val_img_dir = data_dir / "val"

    tf_train = T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    tf_eval = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    print("Loading datasets...")
    train_ds = FairFaceDataset(train_csv, train_img_dir, tf_train, max_samples=args.max_samples)
    val_ds = FairFaceDataset(val_csv, val_img_dir, tf_eval, max_samples=args.max_val_samples)
    print("Train samples:", len(train_ds), " Val samples:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    n_gender = len(train_ds.gender_labels)
    n_age = len(train_ds.age_labels)
    n_race = len(train_ds.race_labels)
    print("Classes -> gender:", n_gender, " age:", n_age, " race:", n_race)

    model = MultiHeadResNet(n_gender, n_age, n_race, pretrained=True).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    best_val = 1e9
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss = train_epoch(model, train_loader, opt, device, crit)
        val_loss = eval_epoch(model, val_loader, device, crit)
        print(f"Train loss: {tr_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            # Save state_dict instead of full model for better portability
            torch.save(model.state_dict(), str(out_path))
            print("Saved best model state_dict to", out_path)

    label_info = {
        "gender_labels": train_ds.gender_labels,
        "age_labels": train_ds.age_labels,
        "race_labels": train_ds.race_labels
    }
    label_path = out_path.parent / "fairface_label_dict.json"
    with open(label_path, "w") as f:
        json.dump(label_info, f, indent=2)
    print("Saved label dict to", label_path)
    print("Training finished.")

if __name__ == "__main__":
    main()
    
