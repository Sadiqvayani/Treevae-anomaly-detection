#!/usr/bin/env python3
"""
Anomaly Detection Experiment for TreeVAE
- Reuses existing TreeVAE code structure exactly
- Trains on single batch excluding anomaly digit
- Tests on training batch + all anomaly samples
- Generates CSV with reconstruction losses and ROC curve
"""

import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
import distutils

# Import existing functions from the project - EXACT same as main.py
from train.train import run_experiment
from utils.utils import prepare_config
from utils.data_utils import get_data
from models.model import TreeVAE
from train.validate_tree import val_tree
from utils.training_utils import get_optimizer, Custom_Metrics
from train.train_tree import run_tree


def get_reconstruction_losses_from_loader(model, data_loader, device):
    """
    Get reconstruction losses for all samples using data loader
    """
    model.eval()
    reconstruction_losses = []
    leaf_assignments = []
    
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            
            # Forward pass using existing model
            outputs = model(batch_data)
            
            # Get leaf assignments using existing output
            if 'p_c_z' in outputs:
                leaf_probs = outputs['p_c_z']
                leaf_assignments.extend(leaf_probs.argmax(dim=-1).cpu().numpy())
            else:
                leaf_assignments.extend(np.zeros(batch_data.size(0)))
            
            # Use the model's forward method to get per-sample losses
            try:
                # Temporarily enable return_elbo to get per-sample losses
                original_return_elbo = model.return_elbo
                model.return_elbo = True
                
                # Get the model's forward pass outputs
                model_outputs = model(batch_data)
                
                # Restore original setting
                model.return_elbo = original_return_elbo
                
                # Extract per-sample reconstruction loss from elbo_samples
                if 'elbo_samples' in model_outputs:
                    # elbo_samples contains: kl_nodes_tot + kl_decisions_tot + kl_root + rec_losses
                    # We need to extract just the rec_losses part
                    elbo_samples = model_outputs['elbo_samples']
                    
                    # For now, use the elbo_samples as a proxy for reconstruction loss
                    # This includes KL terms but should be correlated with reconstruction quality
                    rec_losses = elbo_samples
                else:
                    # Fallback: use the averaged reconstruction loss
                    rec_losses = torch.full((batch_data.size(0),), outputs['rec_loss'].item(), device=device)
                
            except Exception as e:
                print(f"Warning: Could not compute per-sample reconstruction loss: {e}")
                # Fallback: use the averaged reconstruction loss
                rec_losses = torch.full((batch_data.size(0),), outputs['rec_loss'].item(), device=device)
            
            reconstruction_losses.extend(rec_losses.cpu().numpy())
    
    return np.array(reconstruction_losses), np.array(leaf_assignments)


def run_anomaly_detection_experiment(configs, anomaly_digit, device):
    """
    Run complete anomaly detection experiment using EXACT same TreeVAE pipeline
    """
    print("="*60)
    print("TREEVAE ANOMALY DETECTION EXPERIMENT")
    print("="*60)
    print(f"Dataset: {configs['data']['data_name']}")
    print(f"Anomaly digit: {anomaly_digit}")
    print(f"Device: {device}")
    print("="*60)
    
    # Set anomaly digit in config - this will be used by get_data
    configs['data']['anomaly_digit'] = anomaly_digit
    
    # Reduce batch size to 32 for anomaly detection (8 batches of 32 samples)
    # Original batch size is 256, but we reduce to 32 for more batches
    # This creates 8 batches for training with 256 samples
    original_batch_size = configs['training']['batch_size']
    configs['training']['batch_size'] = 32
    print(f"Reduced batch size from {original_batch_size} to {configs['training']['batch_size']} for anomaly detection")
    
    # Get datasets using existing data loading (already handles anomaly detection mode)
    trainset, trainset_eval, testset = get_data(configs)
    
    # For anomaly detection, we need to use ONLY 32 samples for training (overfitting)
    # and create a combined test set with those 32 + all anomaly samples
    
    # Get only 32 samples for training (1 batch for overfitting)
    from torch.utils.data import Subset
    train_indices = trainset.indices[:32]  # Use 32 samples (1 batch of 32)
    trainset = Subset(trainset.dataset, train_indices)  # Override trainset with only 32 samples
    trainset_eval = Subset(trainset.dataset, train_indices)  # Same 32 samples for eval
    
    # Filter test set to only include anomaly digit samples
    # Get the original test dataset to access all test samples
    original_testset = testset.dataset
    anomaly_test_indices = []
    for idx in range(len(original_testset)):
        if original_testset.targets[idx] == anomaly_digit:
            anomaly_test_indices.append(idx)
    
    # Create subset with only anomaly digit samples from test set
    anomaly_testset = Subset(original_testset, anomaly_test_indices)
    
    print(f"Anomaly digit {anomaly_digit} test samples: {len(anomaly_test_indices)}")
    
    # Create combined test set: 32 training samples + only anomaly test samples
    from torch.utils.data import ConcatDataset
    combined_testset = ConcatDataset([trainset, anomaly_testset])
    
    # Create binary labels for anomaly detection
    # Training samples are normal (0), test set samples are anomaly (1)
    anomaly_labels_binary = np.concatenate([
        np.zeros(len(trainset)),  # Normal samples from training (32)
        np.ones(len(anomaly_testset))     # Anomaly samples from test set (only anomaly digit)
    ])
    
    # Create a custom dataset wrapper for val_tree that mimics Subset behavior
    class CombinedTestSet:
        def __init__(self, trainset, anomaly_testset):
            self.trainset = trainset
            self.anomaly_testset = anomaly_testset
            self.dataset = anomaly_testset.dataset  # Use anomaly_testset's dataset as the base
            # Create combined indices: first 32 from trainset, then anomaly test samples
            self.indices = np.concatenate([trainset.indices, anomaly_testset.indices])
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            if idx < len(self.trainset):
                return self.trainset[idx]
            else:
                return self.anomaly_testset[idx - len(self.trainset)]
    
    # Create the combined test set wrapper for val_tree
    combined_testset_for_val = CombinedTestSet(trainset, anomaly_testset)
    
    print(f"Training samples: {len(trainset)} (ONLY 32 for overfitting)")
    print(f"Batch size: {configs['training']['batch_size']} (reduced from {original_batch_size})")
    print(f"Number of training batches: {len(trainset) // configs['training']['batch_size']}")
    print(f"Test samples (anomaly): {len(anomaly_testset)} (only digit {anomaly_digit})")
    print(f"Combined test set: {len(combined_testset)}")
    print(f"Anomaly labels: {np.sum(anomaly_labels_binary)} anomalies out of {len(anomaly_labels_binary)} total")
    
    # Create experiment path for anomaly detection
    from pathlib import Path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = Path(f"models/experiments/{configs['data']['data_name']}_anomaly_digit_{anomaly_digit}_{timestamp}")
    experiment_path.mkdir(parents=True, exist_ok=True)
    configs['experiment_path'] = experiment_path
    
    print(f"Experiment path: {experiment_path}")
    print(f"Anomaly Detection Mode: Training on digits excluding {anomaly_digit}, Testing on all digits including {anomaly_digit}")
    print(f"Anomaly digit: {anomaly_digit}")
    
    # Initialize wandb EXACT same as normal TreeVAE
    import wandb
    import uuid
    import time
    import os
    
    # Set paths
    project_dir = Path(__file__).absolute().parent
    timestr = time.strftime("%Y%m%d-%H%M%S")
    ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
    experiment_path = configs['globals']['results_dir'] / configs['data']['data_name'] / ex_name
    experiment_path.mkdir(parents=True)
    os.makedirs(os.path.join(project_dir, '../models/logs', ex_name))
    print("Experiment path: ", experiment_path)
    
    # Wandb - EXACT same as normal TreeVAE
    os.environ['WANDB_CACHE_DIR'] = os.path.join(project_dir, '../wandb', '.cache', 'wandb')
    os.environ["WANDB_SILENT"] = "true"
    
    # ADD YOUR WANDB ENTITY
    wandb.init(
        project="treevae",
        entity="test",
        config=configs, 
        mode=configs['globals']['wandb_logging']
    )
    
    if configs['globals']['wandb_logging'] in ['online', 'disabled']:
        wandb.run.name = wandb.run.name.split("-")[-1] + "-"+ configs['run_name']
    elif configs['globals']['wandb_logging'] == 'offline':
        wandb.run.name = configs['run_name']
    else:
        raise ValueError('wandb needs to be set to online, offline or disabled.')
    
    # Reproducibility - EXACT same as normal TreeVAE
    from utils.utils import reset_random_seeds
    reset_random_seeds(configs['globals']['seed'])
    
    # Use EXACT same training pipeline as normal TreeVAE
    print("\n" + "="*60)
    print("STARTING FULL TREEVAE TRAINING PIPELINE")
    print("="*60)
    
    # Train using existing run_tree function - EXACT same as normal TreeVAE
    model = run_tree(trainset, trainset_eval, testset, device, configs)
    
    print("\n" + "="*60)
    print("TREEVAE TRAINING COMPLETED - STARTING ANOMALY DETECTION EVALUATION")
    print("="*60)
    
    # Use existing validation function for comprehensive evaluation - EXACT same as normal TreeVAE
    # val_tree expects both datasets to have .dataset and .indices attributes
    val_tree(trainset, combined_testset_for_val, model, device, experiment_path, configs)
    
    # Get reconstruction losses for anomaly detection analysis
    print("\nCalculating reconstruction losses for anomaly detection...")
    
    # Create a data loader for the combined test set
    from utils.data_utils import get_gen
    combined_loader = get_gen(combined_testset, configs, validation=True, shuffle=False)
    
    # Get reconstruction losses using the data loader
    reconstruction_losses, leaf_assignments = get_reconstruction_losses_from_loader(model, combined_loader, device)
    
    # Calculate ROC curve for anomaly detection
    auc_score = roc_auc_score(anomaly_labels_binary, reconstruction_losses)
    fpr, tpr, thresholds = roc_curve(anomaly_labels_binary, reconstruction_losses)
    
    print(f"\nAnomaly Detection AUC: {auc_score:.4f}")
    
    # Create results DataFrame
    # Get true labels from the combined test set
    true_labels = []
    for _, labels in combined_loader:
        true_labels.extend(labels.numpy())
    
    results_df = pd.DataFrame({
        'sample_id': range(len(combined_testset)),
        'true_digit': true_labels,
        'is_anomaly': anomaly_labels_binary.astype(int),
        'reconstruction_loss': reconstruction_losses,
        'leaf_assignment': leaf_assignments
    })
    
    # Create unique output directory for anomaly results with more precise timestamp
    import time
    unique_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    output_dir = f"eval_datasets/anomaly_results_{configs['data']['data_name']}_digit_{anomaly_digit}_{unique_timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results CSV with unique timestamp
    results_file = os.path.join(output_dir, f"anomaly_detection_results_digit_{anomaly_digit}_{unique_timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Results CSV saved to: {results_file}")
    
    # Plot and save ROC curve with unique timestamp
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - Anomaly Detection ({configs["data"]["data_name"]}, Digit {anomaly_digit})', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    roc_file = os.path.join(output_dir, f"roc_curve_digit_{anomaly_digit}_{unique_timestamp}.png")
    plt.savefig(roc_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {roc_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ANOMALY DETECTION EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Dataset: {configs['data']['data_name']}")
    print(f"Anomaly digit: {anomaly_digit}")
    print(f"Training samples: {len(trainset)} (ONLY 32 for overfitting)")
    print(f"Combined test samples: {len(combined_testset)}")
    print(f"  - Normal samples: {len(trainset)} (same 32 as training)")
    print(f"  - Anomaly samples: {len(anomaly_testset)} (only digit {anomaly_digit})")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Results saved to: {output_dir}")
    print(f"TreeVAE experiment path: {experiment_path}")
    print("="*60)
    
    # Show sample results
    print("\nSample results (first 10 rows):")
    print(results_df.head(10))
    
    # Finish wandb - EXACT same as normal TreeVAE
    wandb.finish(quiet=True)
    
    return results_df, auc_score, fpr, tpr, output_dir


def main():
    """
    Main function for anomaly detection experiment
    Reuses existing argument parsing and config preparation - EXACT same as main.py
    """
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)

    parser = argparse.ArgumentParser(description='TreeVAE Anomaly Detection Experiment')

    # Reuse existing arguments from main.py - EXACT same
    parser.add_argument('--data_name', type=str, help='the dataset')
    parser.add_argument('--num_epochs', type=int, help='the number of training epochs')
    parser.add_argument('--num_epochs_finetuning', type=int, help='the number of finetuning epochs')
    parser.add_argument('--num_epochs_intermediate_fulltrain', type=int, help='the number of finetuning epochs during training')
    parser.add_argument('--num_epochs_smalltree', type=int, help='the number of sub-tree training epochs')

    parser.add_argument('--num_clusters_data', type=int, help='the number of clusters in the data')
    parser.add_argument('--num_clusters_tree', type=int, help='the max number of leaves of the tree')

    parser.add_argument('--kl_start', type=float, nargs='?', const=0.,
                        help='initial KL divergence from where annealing starts')
    parser.add_argument('--decay_kl', type=float, help='KL divergence annealing')
    parser.add_argument('--latent_dim', type=str, help='specifies the latent dimensions of the tree')
    parser.add_argument('--mlp_layers', type=str, help='specifies how many layers should the MLPs have')

    parser.add_argument('--grow', type=lambda x: bool(distutils.util.strtobool(x)), help='whether to grow the tree')
    parser.add_argument('--augment', type=lambda x: bool(distutils.util.strtobool(x)), help='augment images or not')
    parser.add_argument('--augmentation_method', type=str, help='none vs simple augmentation vs contrastive approaches')
    parser.add_argument('--aug_decisions_weight', type=float,
                        help='weight of similarity regularizer for augmented images')
    parser.add_argument('--compute_ll', type=lambda x: bool(distutils.util.strtobool(x)),
                        help='whether to compute the log-likelihood')

    # Other parameters
    parser.add_argument('--save_model', type=lambda x: bool(distutils.util.strtobool(x)),
                        help='specifies if the model should be saved')
    parser.add_argument('--eager_mode', type=lambda x: bool(distutils.util.strtobool(x)),
                        help='specifies if the model should be run in graph or eager mode')
    parser.add_argument('--num_workers', type=int, help='number of workers in dataloader')
    parser.add_argument('--seed', type=int, help='random number generator seed')
    parser.add_argument('--wandb_logging', type=str, help='online, disabled, offline enables logging in wandb')

    # Specify config name
    parser.add_argument('--config_name', default='mnist', type=str,
                        choices=['mnist', 'fmnist', 'news20', 'omniglot', 'cifar10', 'cifar100', 'celeba'],
                        help='the override file name for config.yml')
    
    # Anomaly detection specific arguments
    parser.add_argument('--anomaly_digit', type=int, default=2, choices=range(10),
                        help='the digit to use as anomaly (0-9)')

    args = parser.parse_args()
    configs = prepare_config(args, project_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print experiment information - EXACT same format as main.py
    print("="*60)
    print("🔬 TREEVAE ANOMALY DETECTION EXPERIMENT")
    print("="*60)
    print(f"📊 Dataset: {configs['data']['data_name']}")
    print(f"🎯 Anomaly Digit: {args.anomaly_digit}")
    print(f"🌱 Seed: {configs['globals']['seed']}")
    print(f"🔄 Training Epochs: {configs['training']['num_epochs']}")
    print(f"🌱 Small Tree Epochs: {configs['training']['num_epochs_smalltree']}")
    print(f"🔧 Intermediate Full Train Epochs: {configs['training']['num_epochs_intermediate_fulltrain']}")
    print(f"✨ Finetuning Epochs: {configs['training']['num_epochs_finetuning']}")
    print(f"🌳 Tree Clusters: {configs['training']['num_clusters_tree']}")
    print(f"📦 Data Clusters: {configs['data']['num_clusters_data']}")
    print(f"💻 Device: {device}")
    print("="*60)
    print()
    
    # Run anomaly detection experiment
    try:
        results_df, auc_score, fpr, tpr, output_dir = run_anomaly_detection_experiment(
            configs, args.anomaly_digit, device
        )
        
        print(f"\n🎉 Experiment completed successfully!")
        print(f"📊 AUC Score: {auc_score:.4f}")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📄 CSV file: anomaly_detection_results_digit_{args.anomaly_digit}.csv")
        print(f"📈 ROC curve: roc_curve_digit_{args.anomaly_digit}.png")
        
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()