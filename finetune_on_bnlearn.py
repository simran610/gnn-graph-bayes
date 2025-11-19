"""
Fine-tune your trained model on BNLearn graphs
Uses small learning rate to adapt without forgetting
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from graphsage_model import GraphSAGE
from BIF_data_debugging import BenchmarkDatasetProcessor
import glob
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score

def finetune_on_bnlearn(
    pretrained_model_path="training_results/models/graphsage_root_probability_evidence_only_intermediate_logprob_fold_4.pt",
    bif_directory="dataset_bif_files",
    output_path="model_finetuned_bnlearn.pt",
    epochs=50,
    lr=0.00005,  # Very small LR!
    batch_size=8,
    replication_factor=50  # Replicate graphs for more training data
):
    print("="*80)
    print("FINE-TUNING ON BNLEARN GRAPHS")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained model
    model = GraphSAGE(
        in_channels=25,
        hidden_channels=128,
        out_channels=1,
        mode='root_probability',
        use_log_prob=True
    )
    
    model.load_state_dict(torch.load(pretrained_model_path, weights_only=False))
    model = model.to(device)
    
    print(f"✓ Loaded pretrained model from {pretrained_model_path}")
    
    # Load BNLearn graphs with BNLEARN NORMALIZATION
    print("\n" + "="*80)
    print("IMPORTANT: Make sure you've run compute_bnlearn_norm.py first!")
    print("           And updated benchmark_script.py to use bnlearn_norm_stats.pt")
    print("="*80 + "\n")
    
    processor = BenchmarkDatasetProcessor("config.yaml", verbose=True)
    
    # Check normalization
    if processor.norm_stats is None or processor.norm_stats.get('source') != 'bnlearn_graphs':
        print("\n⚠️  WARNING: Not using BNLearn normalization stats!")
        print("   This fine-tuning may not work well.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    bif_files = glob.glob(os.path.join(bif_directory, "*.bif"))
    
    graphs = []
    print(f"\nLoading {len(bif_files)} BNLearn graphs...")
    for bif_path in bif_files:
        network_name = Path(bif_path).stem
        try:
            graph, meta = processor.process_bif_to_graph(bif_path, network_name)
            if graph is not None:
                graphs.append(graph)
        except Exception as e:
            print(f"  ⚠️  Failed {network_name}: {e}")
    
    print(f"✓ Loaded {len(graphs)} graphs")
    
    # Replicate graphs to create more training samples
    replicated_graphs = graphs * replication_factor
    print(f"✓ Replicated to {len(replicated_graphs)} samples (x{replication_factor})")
    
    # Split: 80% train, 20% val
    split_idx = int(0.8 * len(replicated_graphs))
    train_graphs = replicated_graphs[:split_idx]
    val_graphs = replicated_graphs[split_idx:]
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Train: {len(train_graphs)}, Val: {len(val_graphs)}")
    
    # Optimizer with SMALL learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Simple MSE loss (don't use complex loss for fine-tuning)
    loss_fn = torch.nn.MSELoss()
    
    print(f"\n{'='*80}")
    print(f"FINE-TUNING CONFIG:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr} (very small!)")
    print(f"  Batch Size: {batch_size}")
    print(f"  Loss: MSE")
    print(f"{'='*80}\n")
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # ===== TRAIN =====
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            out = model(data)
            target = data.y
            
            loss = loss_fn(out.squeeze(), target.squeeze())
            loss.backward()
            
            # Gradient clipping (important for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # ===== VALIDATE =====
        model.eval()
        val_loss = 0
        all_preds, all_trues = [], []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                target = data.y
                
                loss = loss_fn(out.squeeze(), target.squeeze())
                val_loss += loss.item()
                
                all_preds.extend(out.squeeze().cpu().numpy())
                all_trues.extend(target.squeeze().cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Compute metrics in probability space
        preds_prob = np.exp(np.clip(all_preds, -10, 0))
        trues_prob = np.exp(np.clip(all_trues, -10, 0))
        mae = np.mean(np.abs(preds_prob - trues_prob))
        r2 = r2_score(trues_prob, preds_prob)
        
        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"MAE: {mae:.4f} | "
              f"R²: {r2:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_path)
            print(f"  → Saved best model (val_loss: {avg_val_loss:.4f})")
    
    print(f"\n{'='*80}")
    print("FINE-TUNING COMPLETE!")
    print(f"  Best model saved to: {output_path}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"{'='*80}")
    
    return model

if __name__ == "__main__":
    model = finetune_on_bnlearn(
        epochs=20,
        lr=0.00005,  # Start with this, increase to 0.0001 if loss plateaus
        replication_factor=50
    )
    
    print("\nNEXT STEPS:")
    print("1. Run benchmark with fine-tuned model:")
    print("   Update model_path in benchmark_script.py to 'model_finetuned_bnlearn.pt'")
    print("2. Compare results with original model")