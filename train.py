"""Train an equivariant diffusion model on QM9."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from MolEDM.data import ATOM_TYPES, QM9Dataset
from MolEDM.diffusion import EquivariantDiffusion
from MolEDM.egnn import EGNNDynamics
from MolEDM.stability import evaluate
from MolEDM.utils import EMA, get_device, to_molecules, write_xyz


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/qm9")
    p.add_argument("--out", default="runs/qm9")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-12)
    p.add_argument("--clip-grad", type=float, default=1.5)
    p.add_argument("--ema-decay", type=float, default=0.999)
    # model
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--timesteps", type=int, default=1000)
    # data
    p.add_argument("--remove-h", action="store_true", help="train on heavy atoms only")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=2)
    # eval
    p.add_argument("--eval-every", type=int, default=5, help="epochs between sampling")
    p.add_argument("--n-eval-samples", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build(args, num_types):
    dyn = EGNNDynamics(num_types=num_types, hidden=args.hidden, n_layers=args.layers)
    return EquivariantDiffusion(
        dyn, num_types=num_types, timesteps=args.timesteps
    )


@torch.no_grad()
def val_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x, h, mask = (batch[k].to(device) for k in ("x", "h", "mask"))
        total += model.loss(x, h, mask).item() * len(x)
        n += len(x)
    model.train()
    return total / max(1, n)


@torch.no_grad()
def sample_and_score(model, size_hist, max_n, n_samples, device, symbols, implicit_h=False):
    """Generate molecules and report the standard stability metrics."""
    was_training = model.training
    model.eval()
    sizes = torch.multinomial(size_hist, n_samples, replacement=True).to(device)
    mask = (torch.arange(max_n, device=device)[None, :] < sizes[:, None]).float()
    x, one_hot, _ = model.sample(mask)
    model.train(was_training)
    mols = to_molecules(x, one_hot, mask, symbols)
    return evaluate(mols, implicit_h), mols


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    print(f"device: {device}")

    # Build the datasets and dataloaders
    train_set = QM9Dataset(args.data_root, "train", args.remove_h, args.limit_train)
    val_set = QM9Dataset(args.data_root, "val", args.remove_h, args.limit_val)
    symbols = ATOM_TYPES[1:] if args.remove_h else ATOM_TYPES
    print(f"train {len(train_set)} | val {len(val_set)} | max atoms {train_set.max_n}")

    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.num_workers)

    # Build the mdoel
    model = build(args, train_set.num_types).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e6:.2f}M")

    # Build the optimizer and EMA
    ema = EMA(model, args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            amsgrad=True)

    # Save the config and size histogram for later sampling
    size_hist = train_set.size_histogram()
    meta = dict(vars(args), num_types=train_set.num_types, max_n=train_set.max_n)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log_path = os.path.join(args.out, "log.jsonl")
    best = float("inf")

    # Run the training loop
    for epoch in range(1, args.epochs + 1):
        t0, running, seen = time.time(), 0.0, 0

        # Run one epoch of training passing through the training data
        for i, batch in enumerate(train_loader):

            # Extract the batch 
            x, h, mask = (batch[k].to(device) for k in ("x", "h", "mask"))

            # Compute the loss
            loss = model.loss(x, h, mask)

            # Update the model parameters with backpropagation and EMA
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            ema.update(model)

            running += loss.item()
            seen += 1
            if i % 50 == 0:
                print(f"  ep {epoch} it {i}/{len(train_loader)} "
                      f"loss {running/seen:.4f} |g| {grad_norm:.2f}", flush=True)
                
        # Record the training and validation loss
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "val_loss": val_loss(model, val_loader, device),
            "sec": round(time.time() - t0, 1),
        }

        # If requested, sample molecules and compute the stability metrics
        if args.eval_every and epoch % args.eval_every == 0:
            metrics, mols = sample_and_score(
                ema.shadow, size_hist, train_set.max_n, args.n_eval_samples, device, symbols,
                args.remove_h,
            )
            record.update({f"ema_{k}": v for k, v in metrics.items()})
            for i, (pos, syms) in enumerate(mols[:8]):
                write_xyz(os.path.join(args.out, "samples", f"ep{epoch:04d}_{i}.xyz"),
                          pos, syms, comment=f"epoch {epoch}")

        # Save the log
        print(json.dumps(record), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Checkpoint the model
        ckpt = {
            "model": model.state_dict(),
            "ema": ema.shadow.state_dict(),
            "config": meta,
            "size_hist": size_hist,
            "epoch": epoch,
        }
        torch.save(ckpt, os.path.join(args.out, "last.pt"))

        # Save the best model according to validation loss
        if record["val_loss"] < best:
            best = record["val_loss"]
            torch.save(ckpt, os.path.join(args.out, "best.pt"))

if __name__ == "__main__":
    main()
