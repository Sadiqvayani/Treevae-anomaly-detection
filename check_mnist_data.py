#!/usr/bin/env python3
"""
Script to check MNIST dataset sizes used in TreeVAE
"""
import torch
import torchvision
from torchvision import transforms as T
import numpy as np

# MNIST dataset sizes
print("MNIST Dataset Information:")
print("="*40)

# Full MNIST dataset
full_trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=T.ToTensor())
full_testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=T.ToTensor())

print(f"📊 Full MNIST Dataset:")
print(f"   Training samples: {len(full_trainset):,}")
print(f"   Test samples: {len(full_testset):,}")
print(f"   Total samples: {len(full_trainset) + len(full_testset):,}")

# Check how many samples per digit
print(f"\n📈 Samples per digit in full dataset:")
train_targets = full_trainset.targets.numpy()
test_targets = full_testset.targets.numpy()

for digit in range(10):
    train_count = np.sum(train_targets == digit)
    test_count = np.sum(test_targets == digit)
    print(f"   Digit {digit}: {train_count:,} train + {test_count:,} test = {train_count + test_count:,} total")

# Simulate the select_subset function
print(f"\n🎯 TreeVAE Configuration (from mnist.yml):")
print(f"   num_clusters_data: 10 (uses all 10 digits)")
print(f"   num_clusters_tree: 10")

# Since num_clusters_data = 10, it uses ALL digits (0-9)
# So the actual data used is the full dataset
print(f"\n✅ Actual Data Used by TreeVAE:")
print(f"   Training samples: {len(full_trainset):,}")
print(f"   Test samples: {len(full_testset):,}")
print(f"   Total samples: {len(full_trainset) + len(full_testset):,}")

# Calculate batches
batch_size = 256
train_batches = len(full_trainset) // batch_size
test_batches = len(full_testset) // batch_size

print(f"\n📦 Batch Information:")
print(f"   Batch size: {batch_size}")
print(f"   Training batches: {train_batches:,}")
print(f"   Test batches: {test_batches:,}")
print(f"   Training samples per batch: ~{len(full_trainset) // train_batches}")
print(f"   Test samples per batch: ~{len(full_testset) // test_batches}")
