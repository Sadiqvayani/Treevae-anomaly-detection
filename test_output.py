#!/usr/bin/env python3
"""
Test script to show the new output format
"""
import yaml

# Load the config to show what the output will look like
with open('configs/mnist.yml', 'r') as f:
    configs = yaml.safe_load(f)

# Print experiment information at the top
print("="*60)
print("🔬 TREEVAE EXPERIMENT CONFIGURATION")
print("="*60)
print(f"📊 Dataset: {configs['data']['data_name']}")
print(f"🎯 Seed: {configs['globals']['seed']}")
print(f"🔄 Training Epochs: {configs['training']['num_epochs']}")
print(f"🌱 Small Tree Epochs: {configs['training']['num_epochs_smalltree']}")
print(f"🔧 Intermediate Full Train Epochs: {configs['training']['num_epochs_intermediate_fulltrain']}")
print(f"✨ Finetuning Epochs: {configs['training']['num_epochs_finetuning']}")
print(f"📈 Total Epochs: {configs['training']['num_epochs'] + configs['training']['num_epochs_smalltree'] + configs['training']['num_epochs_intermediate_fulltrain'] + configs['training']['num_epochs_finetuning']}")
print(f"🌳 Tree Clusters: {configs['training']['num_clusters_tree']}")
print(f"📦 Data Clusters: {configs['data']['num_clusters_data']}")
print("="*60)
print()
print("Project directory: /scratch/bep91wab/projectTreeVAE_Pytorch/treevae")
print("Using Tesla V100-PCIE-32GB")
print("Experiment path:  /scratch/bep91wab/projectTreeVAE_Pytorch/treevae/models/experiments/mnist/20250723-130407_b6679")
print("Epoch 0, Train     : loss_value: 230.868 rec_loss: 230.868 kl_decisions: 0.628 kl_root: 24.674 kl_nodes: 49.351 aug_decisions: 0.000 perc_samples: 0.518 nmi: 0.160 accuracy: 0.186 alpha: 0.000")
print("...")
print("🔁 Running experiment with seed from config: 17")
