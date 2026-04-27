import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from model import GPT, GPTConfig

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_data(p, op):
    """Generates a o b = c mod p data."""
    pairs = [(a, b) for a in range(p) for b in range(p)]
    if op == '/': pairs = [(a, b) for a, b in pairs if b != 0]
    
    data = []
    for a, b in pairs:
        if op == '+': c = (a + b) % p
        elif op == '-': c = (a - b) % p
        elif op == '/': c = (a * pow(b, -1, p)) % p
        
        # Token mapping: 0..p-1 are numbers. p=BOS, p+1='+', p+2='-', p+3='/', p+4='=', p+5=EOS
        op_idx = p+1 if op=='+' else p+2 if op=='-' else p+3
        data.append([p, a, op_idx, b, p+4, c, p+5])
    
    random.shuffle(data)
    split = int(len(data) * 0.8)
    return torch.tensor(data[:split]), torch.tensor(data[split:])

def run_experiment(p=97, op='+', n_layer=1, max_steps=10000, wd=0.1, name="exp"):
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_data, val_data = generate_data(p, op)
    
    config = GPTConfig(
        block_size = 8,
        vocab_size = p + 6,
        n_layer = n_layer,
        n_head = 4,
        n_embd = 128,
        dropout = 0.0
    )
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(weight_decay=wd, learning_rate=1e-3, betas=(0.9, 0.98), device_type=device)
    
    history = {"step": [], "train_loss": [], "train_acc": [], "val_acc": []}

    for step in range(max_steps + 1):
        model.train()
        idx = torch.randint(0, len(train_data), (512,))
        batch = train_data[idx].to(device)
        
        logits = model(batch[:, :-1])
        # Loss only on the answer token 'c' which is index 5 in sequence
        # The prediction for index 5 is at logits index 4
        loss = F.cross_entropy(logits[:, 4, :], batch[:, 5])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                # Train Acc
                train_preds = torch.argmax(logits[:, 4, :], dim=-1)
                t_acc = (train_preds == batch[:, 5]).float().mean().item()
                
                # Val Acc
                v_batch = val_data.to(device)
                v_logits = model(v_batch[:, :-1])
                v_preds = torch.argmax(v_logits[:, 4, :], dim=-1)
                v_acc = (v_preds == v_batch[:, 5]).float().mean().item()
                
                history["step"].append(step)
                history["train_loss"].append(loss.item())
                history["train_acc"].append(t_acc)
                history["val_acc"].append(v_acc)
                print(f"[{name}] Step {step} | Loss: {loss.item():.4f} | Val Acc: {v_acc:.4f}")

    # Save Checkpoint
    torch.save({'model_state': model.state_dict(), 'config': config, 'p': p}, f"{name}.pt")
    return history

# --- Helper for Plotting ---
def save_plot(history, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(history["step"], history["train_acc"], label="Train Accuracy", alpha=0.7)
    plt.plot(history["step"], history["val_acc"], label="Test Accuracy", lw=2)
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(filename)
    plt.close()

# --- The Updated Runner ---
def run_all_deliverables():
    # Ensure a directory exists for outputs
    os.makedirs("deliverables", exist_ok=True)

    print("Running 0.1 Sanity Check...")
    # Using p=10 to keep vocab small. Sequence: [BOS, 1, 2, 3, 4, EOS]
    sanity_data = torch.tensor([[10, 1, 2, 3, 4, 15]]) 
    # Run a tiny training session
    run_experiment(p=10, op='+', max_steps=500, wd=0.0, name="sanity_check")
    
    # 1.1 DATA GENERATION LOGS
    print("\n--- Generating 1.1 Deliverables (Data Splits) ---")
    for p_val in [97, 113]:
        train_d, val_d = generate_data(p_val, '+')
        print(f"p={p_val}: Total Pairs: {p_val**2} | Train: {len(train_d)} | Val: {len(val_d)}")

    # 1.2 WARMUP: ADDITION & SUBTRACTION
    print("\n--- Generating 1.2 Deliverables (Warmup Curves) ---")
    # p=97 Addition, 1-layer, 3 random seeds
    warmup_seeds = [42, 101, 2024]
    plt.figure(figsize=(10, 5))
    for s in warmup_seeds:
        seed_everything(s)
        hist = run_experiment(p=97, op='+', n_layer=1, max_steps=5000, wd=0.1, name=f"warmup_s{s}")
        plt.plot(hist["step"], hist["val_acc"], label=f"Seed {s} Test Acc")
    
    plt.title("1.2 Warmup: Addition p=97 (3 Random Restarts)")
    plt.legend()
    plt.savefig("deliverables/1.2_warmup_seeds.png")
    plt.close()

    # 1.3 GROKKING: DIVISION
    print("\n--- Generating 1.3 Deliverables (Grokking Plot) ---")
    # Higher WD and steps to ensure grokking happens
    grok_hist = run_experiment(p=97, op='/', n_layer=1, max_steps=30000, wd=0.8, name="grokking_final")
    save_plot(grok_hist, "1.3 Grokking Modular Division (p=97)", "deliverables/1.3_grokking_division.png")

    # 1.4 ABLATION STUDIES
    print("\n--- Generating 1.4 Deliverables (Ablation Plots) ---")
    
    # Ablation 1: Low Weight Decay (Should fail to grok or grok very slowly)
    print("Ablation: Low Weight Decay...")
    abl1_hist = run_experiment(p=97, op='/', n_layer=1, max_steps=20000, wd=0.01, name="ablation_low_wd")
    save_plot(abl1_hist, "1.4 Ablation: Low Weight Decay (0.01)", "deliverables/1.4_ablation_wd.png")

    # Ablation 2: Different Architecture (2-Layers)
    print("Ablation: 2-Layer Model...")
    abl2_hist = run_experiment(p=97, op='/', n_layer=2, max_steps=20000, wd=0.5, name="ablation_2layer")
    save_plot(abl2_hist, "1.4 Ablation: 2-Layer Architecture", "deliverables/1.4_ablation_layers.png")

    print("\nAll deliverables generated in the 'deliverables/' folder.")

if __name__ == "__main__":
    run_all_deliverables()