import torch
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import os
from src.utils.config import JEPAConfig
from src.models.jepa import MiniIJEPA
from src.training.metrics import linear_probe

def parse_log(log_path="training_log.txt"):
    epoch_losses = {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "->" in line and "Epoch" in line and "loss=" in line and "batch" not in line:
                try:
                    parts = line.strip()
                    epoch = int(parts.split("Epoch")[1].split("|")[0].strip())
                    loss = float(parts.split("loss=")[1].strip())
                    epoch_losses[epoch] = loss
                except:
                    continue
    return epoch_losses
def plot_loss(epoch_losses, save_path="outputs/loss_curve.png"):
    os.makedirs("outputs", exist_ok=True)
    epochs = sorted(epoch_losses.keys())
    losses = [epoch_losses[e] for e in epochs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, losses, linewidth=2, color="#4C72B0", marker="o", markersize=3)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("L2 Loss", fontsize=12)
    ax.set_title("Mini I-JEPA - Training Loss", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved -> {save_path}")

def main():
    # ── parse log ─────────────────────────────────────────────────────
    print("Parsing training log...")
    epoch_losses = parse_log()
    if not epoch_losses:
        print("No epoch summaries found in log yet.")
        return

    best_epoch = min(epoch_losses, key=epoch_losses.get)
    print(f"Epochs completed : {len(epoch_losses)}")
    print(f"Best loss        : {epoch_losses[best_epoch]:.4f} at epoch {best_epoch}")
    print(f"Final loss       : {epoch_losses[max(epoch_losses)]:.4f}")

    # ── plot loss curve ───────────────────────────────────────────────
    plot_loss(epoch_losses)

    # ── load best model ───────────────────────────────────────────────
    cfg = JEPAConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniIJEPA(cfg)
    checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device)
    print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")

    # ── linear probe ──────────────────────────────────────────────────
    print("Running linear probe on STL-10 train split...")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    labelled = torchvision.datasets.STL10(
        root="data/", split="train",
        transform=transform, download=True
    )
    loader = torch.utils.data.DataLoader(
        labelled, batch_size=64,
        shuffle=True, num_workers=0
    )
    acc = linear_probe(model.context_encoder, loader, device, num_classes=10)
    print(f"Linear probe accuracy: {acc*100:.1f}%")

    # ── attention map ─────────────────────────────────────────────────
    from src.utils.visualize import visualize_attention
    from src.data.dataset import STL10JEPADataset
    dataset = STL10JEPADataset(cfg, split="train")
    image, _, _ = dataset[0]
    visualize_attention(model, image, cfg, save_path="outputs/attention_map.png")

    # ── final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Epochs trained   : {len(epoch_losses)}")
    print(f"Best loss        : {epoch_losses[best_epoch]:.4f} (epoch {best_epoch})")
    print(f"Linear probe acc : {acc*100:.1f}%")
    print(f"Loss curve       : outputs/loss_curve.png")
    print(f"Attention map    : outputs/attention_map.png")
    print("=" * 50)
    print("\nPaste these numbers into your README results table.")

if __name__ == "__main__":
    main()