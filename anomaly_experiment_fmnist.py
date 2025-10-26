#!/usr/bin/env python3
"""
Anomaly Detection Experiment for TreeVAE with Fashion-MNIST augmentation
- Reuses existing TreeVAE code structure exactly
- Trains on single batch excluding anomaly digit
- Tests on training batch + all anomaly samples + Fashion-MNIST samples
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
import torchvision
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader, Subset, ConcatDataset

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
                
                # Extract per-sample reconstruction loss (pure reconstruction, no KL terms)
                if 'rec_loss_samples' in model_outputs:
                    # Use pure reconstruction loss samples (no KL terms)
                    rec_losses = model_outputs['rec_loss_samples']
                elif 'elbo_samples' in model_outputs:
                    # Fallback: if rec_loss_samples not available, use elbo_samples
                    # elbo_samples contains: kl_nodes_tot + kl_decisions_tot + kl_root + rec_losses
                    elbo_samples = model_outputs['elbo_samples']
                    rec_losses = elbo_samples
                else:
                    # Final fallback: use the averaged reconstruction loss
                    rec_losses = torch.full((batch_data.size(0),), outputs['rec_loss'].item(), device=device)
                
            except Exception as e:
                print(f"Warning: Could not compute per-sample reconstruction loss: {e}")
                # Fallback: use the averaged reconstruction loss
                rec_losses = torch.full((batch_data.size(0),), outputs['rec_loss'].item(), device=device)
            
            reconstruction_losses.extend(rec_losses.cpu().numpy())
    
    return np.array(reconstruction_losses), np.array(leaf_assignments)


def create_combined_testset_with_fmnist(configs, anomaly_digit):
    """
    Create combined test set with MNIST + Fashion-MNIST
    - MNIST test set (10,000 samples): 9 normal digits + 1 anomaly digit
    - Fashion-MNIST test set (1,000 samples): all treated as anomaly
    - Total: 11,000 samples with 2,000 anomalies (18.2% anomaly rate)
    """
    print("Creating combined test set with Fashion-MNIST...")
    
    # Get MNIST test set (normal way)
    mnist_configs = configs.copy()
    mnist_configs['data']['anomaly_digit'] = anomaly_digit
    mnist_trainset, mnist_trainset_eval, mnist_testset = get_data(mnist_configs)
    
    # Load Fashion-MNIST test set
    data_path = './data/'
    fmnist_testset = torchvision.datasets.FashionMNIST(
        root=data_path, 
        train=False, 
        download=True, 
        transform=T.ToTensor()
    )
    
    # Take 1,000 samples from Fashion-MNIST (evenly distributed: 100 samples per class)
    np.random.seed(configs['globals']['seed'])
    fmnist_indices = []
    
    # Fashion-MNIST has 10 classes with 1,000 test samples each
    # Take 100 samples from each class for even distribution
    for class_label in range(10):
        class_indices = np.where(fmnist_testset.targets == class_label)[0]
        selected_indices = np.random.choice(class_indices, 100, replace=False)
        fmnist_indices.extend(selected_indices)
    
    fmnist_indices = np.array(fmnist_indices)
    fmnist_subset = Subset(fmnist_testset, fmnist_indices)
    
    print(f"MNIST test samples: {len(mnist_testset)}")
    print(f"Fashion-MNIST test samples: {len(fmnist_subset)}")
    print(f"Total test samples: {len(mnist_testset) + len(fmnist_subset)}")
    
    # Verify even distribution of Fashion-MNIST classes
    fmnist_class_counts = {}
    for idx in fmnist_indices:
        class_label = fmnist_testset.targets[idx].item()
        fmnist_class_counts[class_label] = fmnist_class_counts.get(class_label, 0) + 1
    
    print(f"Fashion-MNIST class distribution: {dict(sorted(fmnist_class_counts.items()))}")
    print(f"Expected: 100 samples per class (0-9)")
    
    # Combine datasets
    combined_testset = ConcatDataset([mnist_testset, fmnist_subset])
    
    # Create a wrapper to make ConcatDataset compatible with TreeVAE training
    class ConcatDatasetWrapper:
        def __init__(self, concat_dataset, mnist_testset, fmnist_subset):
            self.concat_dataset = concat_dataset
            self.mnist_testset = mnist_testset
            self.fmnist_subset = fmnist_subset
            
        def __len__(self):
            return len(self.concat_dataset)
            
        def __getitem__(self, idx):
            return self.concat_dataset[idx]
            
        # Add attributes that TreeVAE training expects
        @property
        def dataset(self):
            # Return a mock dataset with targets attribute
            class MockDataset:
                def __init__(self, mnist_testset, fmnist_subset):
                    # Combine targets from both datasets
                    mnist_targets = []
                    for _, labels in mnist_testset:
                        mnist_targets.extend(labels.numpy() if hasattr(labels, 'numpy') else [labels])
                    
                    fmnist_targets = []
                    for idx in range(len(fmnist_subset)):
                        _, label = fmnist_subset[idx]
                        fmnist_targets.append(label.item() if hasattr(label, 'item') else label)
                    
                    self.targets = torch.tensor(mnist_targets + fmnist_targets)
            
            return MockDataset(self.mnist_testset, self.fmnist_subset)
        
        @property
        def indices(self):
            # Return indices for the combined dataset
            return list(range(len(self.concat_dataset)))
    
    # Wrap the combined dataset
    combined_testset = ConcatDatasetWrapper(combined_testset, mnist_testset, fmnist_subset)
    
    # Create labels for the combined dataset
    # MNIST labels: 0 for normal digits, 1 for anomaly digit
    mnist_labels = []
    for _, labels in mnist_testset:
        mnist_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else [labels])
    mnist_labels = np.array(mnist_labels)
    mnist_binary_labels = (mnist_labels == anomaly_digit).astype(int)
    
    # Fashion-MNIST labels: all 1 (anomaly)
    fmnist_binary_labels = np.ones(len(fmnist_subset), dtype=int)
    
    # Combine binary labels
    combined_binary_labels = np.concatenate([mnist_binary_labels, fmnist_binary_labels])
    
    # Create true labels for reference (MNIST digits + Fashion-MNIST classes)
    # Get actual Fashion-MNIST class labels for the selected samples
    fmnist_true_labels = []
    for idx in fmnist_indices:
        fmnist_true_labels.append(fmnist_testset.targets[idx].item())
    fmnist_true_labels = np.array(fmnist_true_labels)
    
    combined_true_labels = np.concatenate([mnist_labels, fmnist_true_labels])
    
    print(f"Normal samples (MNIST): {np.sum(mnist_binary_labels == 0)}")
    print(f"Anomaly samples (MNIST digit {anomaly_digit}): {np.sum(mnist_binary_labels == 1)}")
    print(f"Anomaly samples (Fashion-MNIST): {len(fmnist_subset)}")
    print(f"Total anomaly samples: {np.sum(combined_binary_labels)}")
    print(f"Anomaly rate: {np.sum(combined_binary_labels) / len(combined_binary_labels) * 100:.1f}%")
    
    return combined_testset, combined_binary_labels, combined_true_labels, mnist_trainset, mnist_trainset_eval


def run_anomaly_detection_experiment_with_fmnist(configs, anomaly_digit, device):
    """
    Run complete anomaly detection experiment with Fashion-MNIST augmentation
    """
    print("="*60)
    print("TREEVAE ANOMALY DETECTION EXPERIMENT WITH FASHION-MNIST")
    print("="*60)
    print(f"Dataset: {configs['data']['data_name']} + Fashion-MNIST")
    print(f"Anomaly digit: {anomaly_digit}")
    print(f"Device: {device}")
    print("="*60)
    
    # Set anomaly digit in config
    configs['data']['anomaly_digit'] = anomaly_digit
    
    # Create combined test set with Fashion-MNIST
    combined_testset, anomaly_labels_binary, true_labels, trainset, trainset_eval = create_combined_testset_with_fmnist(configs, anomaly_digit)
    
    print(f"Training samples: {len(trainset)} (9 MNIST digits excluding {anomaly_digit})")
    print(f"Test samples: {len(combined_testset)} (9k MNIST normal + 1k MNIST anomaly + 1k Fashion-MNIST)")
    print(f"  - Normal samples: {np.sum(anomaly_labels_binary == 0)} (9 MNIST digits)")
    print(f"  - Anomaly samples: {np.sum(anomaly_labels_binary == 1)} (1k MNIST digit {anomaly_digit} + 1k Fashion-MNIST)")
    
    # Create experiment path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = Path(f"models/experiments/{configs['data']['data_name']}_fmnist_anomaly_digit_{anomaly_digit}_{timestamp}")
    print(f"Experiment path: {experiment_path}")
    experiment_path.mkdir(parents=True, exist_ok=True)
    configs['experiment_path'] = experiment_path
    
    # Initialize wandb EXACT same as normal TreeVAE
    import wandb
    import uuid
    import time
    import os
    
    # Set paths
    project_dir = Path(__file__).absolute().parent
    timestr = time.strftime("%Y%m%d-%H%M%S")
    ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
    experiment_path = configs['globals']['results_dir'] / f"{configs['data']['data_name']}_fmnist" / ex_name
    experiment_path.mkdir(parents=True)
    os.makedirs(os.path.join(project_dir, '../models/logs', ex_name))
    print("Experiment path: ", experiment_path)
    
    # Wandb - EXACT same as normal TreeVAE
    os.environ['WANDB_CACHE_DIR'] = os.path.join(project_dir, '../wandb', '.cache', 'wandb')
    os.environ["WANDB_SILENT"] = "true"
    
    # ADD YOUR WANDB ENTITY
    wandb.init(
        project="treevae-fmnist-anomaly",
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
    model = run_tree(trainset, trainset_eval, combined_testset, device, configs)
    
    print("\n" + "="*60)
    print("TREEVAE TRAINING COMPLETED - STARTING ANOMALY DETECTION EVALUATION")
    print("="*60)
    
    # Use existing validation function for comprehensive evaluation
    val_tree(trainset, combined_testset, model, device, experiment_path, configs)
    
    # Get reconstruction losses for analysis
    print("\nCalculating reconstruction losses for anomaly detection...")
    
    # Create a data loader for the combined test set
    from utils.data_utils import get_gen
    combined_loader = get_gen(combined_testset, configs, validation=True, shuffle=False)
    
    # Get reconstruction losses using the data loader
    reconstruction_losses, leaf_assignments = get_reconstruction_losses_from_loader(model, combined_loader, device)
    
    # Calculate ROC curve and AUC
    auc_score = roc_auc_score(anomaly_labels_binary, reconstruction_losses)
    fpr, tpr, thresholds = roc_curve(anomaly_labels_binary, reconstruction_losses)
    
    print(f"\nAnomaly Detection AUC: {auc_score:.4f}")
    
    # Create results DataFrame
    # Create dataset source and class labels
    dataset_sources = ['MNIST'] * (len(combined_testset) - 1000) + ['Fashion-MNIST'] * 1000
    class_labels = []
    
    # MNIST class labels (0-9)
    for i in range(len(combined_testset) - 1000):
        class_labels.append(f"MNIST_{true_labels[i]}")
    
    # Fashion-MNIST class labels (0-9)
    for i in range(len(combined_testset) - 1000, len(combined_testset)):
        class_labels.append(f"FMNIST_{true_labels[i]}")
    
    results_df = pd.DataFrame({
        'sample_id': range(len(combined_testset)),
        'true_digit': true_labels,
        'is_anomaly': anomaly_labels_binary.astype(int),
        'reconstruction_loss': reconstruction_losses,
        'leaf_assignment': leaf_assignments,
        'dataset_source': dataset_sources,
        'class_label': class_labels
    })
    
    # Create unique output directory for results
    unique_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    seed = configs['globals']['seed']
    
    output_dir = f"eval_datasets/fmnist_anomaly_{anomaly_digit}_seed_{seed}_{unique_timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results CSV
    results_file = os.path.join(output_dir, f"fmnist_anomaly_{anomaly_digit}_seed_{seed}_{unique_timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Results CSV saved to: {results_file}")
    
    # Plot and save ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - Anomaly Detection with Fashion-MNIST\n({configs["data"]["data_name"]}, Digit {anomaly_digit}, Seed {seed})', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    roc_file = os.path.join(output_dir, f"roc_curve_fmnist_anomaly_{anomaly_digit}_seed_{seed}_{unique_timestamp}.png")
    plt.savefig(roc_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {roc_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ANOMALY DETECTION EXPERIMENT WITH FASHION-MNIST SUMMARY")
    print("="*60)
    print(f"Dataset: {configs['data']['data_name']} + Fashion-MNIST")
    print(f"Anomaly digit: {anomaly_digit}")
    print(f"Seed: {seed}")
    print(f"Training samples: {len(trainset)} (9 MNIST digits excluding {anomaly_digit})")
    print(f"Test samples: {len(combined_testset)} (9k MNIST normal + 1k MNIST anomaly + 1k Fashion-MNIST)")
    print(f"  - Normal samples: {np.sum(anomaly_labels_binary == 0)} (9 MNIST digits)")
    print(f"  - Anomaly samples: {np.sum(anomaly_labels_binary == 1)} (1k MNIST digit {anomaly_digit} + 1k Fashion-MNIST)")
    print(f"Anomaly rate: {np.sum(anomaly_labels_binary) / len(anomaly_labels_binary) * 100:.1f}%")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Results saved to: {output_dir}")
    print(f"TreeVAE experiment path: {experiment_path}")
    print("="*60)
    
    # Show sample results
    print("\nSample results (first 10 rows):")
    print(results_df.head(10))
    
    # Show breakdown by dataset source
    print("\nBreakdown by dataset source:")
    print(results_df.groupby('dataset_source')['is_anomaly'].value_counts())
    
    # Show Fashion-MNIST class distribution
    print("\nFashion-MNIST class distribution:")
    fmnist_df = results_df[results_df['dataset_source'] == 'Fashion-MNIST']
    print(fmnist_df['class_label'].value_counts().sort_index())
    
    # Finish wandb
    wandb.finish(quiet=True)
    
    return results_df, auc_score, fpr, tpr, output_dir


def main():
    """
    Main function for anomaly detection experiment with Fashion-MNIST
    """
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)

    parser = argparse.ArgumentParser(description='TreeVAE Anomaly Detection Experiment with Fashion-MNIST')

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
    parser.add_argument('--anomaly_digit', type=int, default=0, choices=range(10),
                        help='the digit to use as anomaly (0-9)')

    args = parser.parse_args()
    configs = prepare_config(args, project_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print experiment information
    print("="*60)
    print("🔬 TREEVAE ANOMALY DETECTION WITH FASHION-MNIST")
    print("="*60)
    print(f"📊 Dataset: {configs['data']['data_name']} + Fashion-MNIST")
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
    
    # Run experiment
    try:
        results_df, auc_score, fpr, tpr, output_dir = run_anomaly_detection_experiment_with_fmnist(
            configs, args.anomaly_digit, device
        )
        
        print(f"\n🎉 Experiment completed successfully!")
        print(f"📊 AUC Score: {auc_score:.4f}")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📄 CSV file: fmnist_anomaly_{args.anomaly_digit}_seed_{configs['globals']['seed']}_*.csv")
        print(f"📈 ROC curve: roc_curve_fmnist_anomaly_{args.anomaly_digit}_seed_{configs['globals']['seed']}_*.png")
        
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
