#!/usr/bin/env python3
"""
Detailed Probability Analysis for TreeVAE
- Captures routing probabilities at each node and depth
- Exports comprehensive Excel output with all probabilities
- Includes reconstruction losses and final sample losses
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
from tqdm import tqdm

# Import existing functions from the project
from train.train import run_experiment
from utils.utils import prepare_config
from utils.data_utils import get_data
from models.model import TreeVAE
from train.validate_tree import val_tree
from utils.training_utils import get_optimizer, Custom_Metrics
from train.train_tree import run_tree


def get_detailed_probabilities_from_loader(model, data_loader, device, anomaly_labels_binary, true_labels):
    """
    Get detailed probabilities for all samples using data loader
    Uses the EXACT same method as anomaly_experiment_fmnist.py for reconstruction losses
    """   
    model.eval()
    all_sample_data = []
    all_node_data = []
    all_leaf_data = []
    reconstruction_losses = []
    leaf_assignments = []
    
    printed_mismatch_count = 0
    with torch.no_grad():
        for batch_idx, (batch_data, batch_labels) in enumerate(tqdm(data_loader, desc="Processing batches")):
            batch_data = batch_data.to(device)
            
            # FIRST: Single forward with return_elbo=True to mirror anomaly_experiment and get per-sample losses
            original_return_elbo = model.return_elbo
            model.return_elbo = True
            outputs = model(batch_data)
            model.return_elbo = original_return_elbo
            
            # Get leaf assignments using existing output (same as anomaly_experiment_fmnist.py)
            if 'p_c_z' in outputs:
                leaf_probs = outputs['p_c_z']
                batch_leaf_assignments = leaf_probs.argmax(dim=-1).cpu().numpy()
            else:
                batch_leaf_assignments = np.zeros(batch_data.size(0))
            
            # Per-sample reconstruction losses from this same forward
            if 'rec_loss_samples' in outputs:
                rec_losses = outputs['rec_loss_samples']
            elif 'elbo_samples' in outputs:
                rec_losses = outputs['elbo_samples']
            else:
                rec_losses = torch.full((batch_data.size(0),), outputs['rec_loss'].item(), device=device)
            
            reconstruction_losses.extend(rec_losses.cpu().numpy())
            leaf_assignments.extend(batch_leaf_assignments)
            
            # NOW: Process each sample in the batch individually for detailed analysis
            for sample_idx in range(batch_data.size(0)):
                sample_id = batch_idx * data_loader.batch_size + sample_idx
                single_sample = batch_data[sample_idx:sample_idx+1]  # Keep batch dimension
                single_label = batch_labels[sample_idx]
                
                # Extract sample-specific outputs from batch forward to avoid re-sampling z
                # This ensures weighted_sum matches final_rec_loss exactly
                sample_p_c_z = outputs['p_c_z'][sample_idx:sample_idx+1]  # shape [1, num_leaves]
                sample_node_leaves = outputs['node_leaves']  # list of dicts (same for all samples in batch)
                
                # Get detailed outputs for this sample using batch forward results
                sample_outputs = get_single_sample_detailed_outputs(
                    model, single_sample, device, 
                    batch_outputs=outputs, 
                    sample_idx_in_batch=sample_idx
                )
                
                # Use the EXACT reconstruction loss from this batch forward
                standard_rec_loss = rec_losses[sample_idx].item()
                standard_leaf_assignment = leaf_assignments[sample_id]
                
                # Determine dataset source and class label
                if sample_id < len(true_labels):
                    true_digit = true_labels[sample_id]
                    is_anomaly = anomaly_labels_binary[sample_id]
                    
                    if sample_id < 10000:  # MNIST samples
                        dataset_source = 'MNIST'
                        class_label = f'MNIST_{true_digit}'
                    else:  # Fashion-MNIST samples
                        dataset_source = 'Fashion-MNIST'
                        class_label = f'FMNIST_{true_digit}'
                else:
                    true_digit = single_label.item()
                    is_anomaly = 0
                    dataset_source = 'MNIST'
                    class_label = f'MNIST_{true_digit}'
                
                # Optional: print at most 50 mismatches to logs
                if 'weighted_sum' in sample_outputs:
                    diff_val = abs(sample_outputs['weighted_sum'] - standard_rec_loss)
                    if diff_val > 0.1 and printed_mismatch_count < 50:
                        print(f"Warning: sample_id={sample_id} weighted_sum={sample_outputs['weighted_sum']:.6f} != final_rec_loss={standard_rec_loss:.6f}, diff={diff_val:.6f}")
                        printed_mismatch_count += 1

                # Store sample-level data with EXACT same reconstruction loss and leaf assignment
                sample_data = {
                    'sample_id': sample_id,
                    'true_digit': true_digit,
                    'is_anomaly': is_anomaly,
                    'final_rec_loss': standard_rec_loss,  # Use EXACT same reconstruction loss
                    'leaf_assignment': standard_leaf_assignment,  # Use EXACT same leaf assignment
                    'dataset_source': dataset_source,
                    'class_label': class_label,
                    'num_nodes_processed': len(sample_outputs['node_data']),
                    'num_leaves': len(sample_outputs['leaf_data']),
                    'weighted_sum_leaf_losses': sample_outputs.get('weighted_sum', None),
                    'weighted_sum_diff': (abs(sample_outputs['weighted_sum'] - standard_rec_loss)
                                          if 'weighted_sum' in sample_outputs else None)
                }
                all_sample_data.append(sample_data)
                
                # Store node-level data
                for node_info in sample_outputs['node_data']:
                    node_info['sample_id'] = sample_id
                    node_info['true_digit'] = true_digit
                    node_info['is_anomaly'] = is_anomaly
                    node_info['dataset_source'] = dataset_source
                    node_info['class_label'] = class_label
                    all_node_data.append(node_info)
                
                # Store leaf-level data
                for leaf_info in sample_outputs['leaf_data']:
                    leaf_info['sample_id'] = sample_id
                    leaf_info['true_digit'] = true_digit
                    leaf_info['is_anomaly'] = is_anomaly
                    leaf_info['dataset_source'] = dataset_source
                    leaf_info['class_label'] = class_label
                    all_leaf_data.append(leaf_info)
    
    return all_sample_data, all_node_data, all_leaf_data


def get_single_sample_detailed_outputs(model, x, device, batch_outputs=None, sample_idx_in_batch=0):
    """
    Get detailed outputs for a single sample including all routing probabilities
    Uses batch forward outputs if provided to avoid re-sampling z and ensure exact match with final_rec_loss
    """
    epsilon = 1e-7
    
    if batch_outputs is not None:
        # Reuse batch forward outputs to avoid re-sampling z
        standard_outputs = batch_outputs
        # Extract sample-specific p_c_z from batch
        p_c_z = batch_outputs['p_c_z'][sample_idx_in_batch:sample_idx_in_batch+1]  # shape [1, num_leaves]
        node_leaves = batch_outputs['node_leaves']  # same for all samples in batch
        standard_rec_loss = None  # Will use rec_loss_samples from batch instead
    else:
        # Fallback: do a separate forward (will cause slight mismatch due to re-sampling)
        with torch.no_grad():
            standard_outputs = model(x)
            standard_rec_loss = standard_outputs['rec_loss'].item()
            p_c_z = standard_outputs['p_c_z']  # shape [1, num_leaves]
            node_leaves = standard_outputs['node_leaves']  # list of dicts with 'prob' and 'z_sample' per leaf
    
    # Build leaf-level data WITHOUT re-sampling: reuse the exact z_sample/prob from standard_outputs
    node_data = []
    leaf_data = []
    leaves_prob = []
    reconstructions = []

    # Get leaves in left-to-right order to access their decoders and depth
    leaves_list = model.compute_leaves()  # list of {'node': leaf_node, 'depth': depth}

    num_leaves = p_c_z.size(-1)
    for i in range(num_leaves):
        leaf_node = leaves_list[i]['node']
        depth_level = leaves_list[i]['depth']
        prob_i = p_c_z[:, i]  # tensor shape [1]
        
        # Extract sample-specific z_sample from batch results
        if batch_outputs is not None:
            # node_leaves[i]['z_sample'] has batch dimension [batch_size, latent_dim], extract specific sample
            z_sample_batch = node_leaves[i]['z_sample']
            if z_sample_batch.dim() > 1:
                # Extract the z_sample for this specific sample in the batch
                if z_sample_batch.size(0) > 1:
                    # Batch dimension exists, extract sample-specific z_sample
                    z_sample_i = z_sample_batch[sample_idx_in_batch:sample_idx_in_batch+1]  # Keep batch dim for decoder
                else:
                    z_sample_i = z_sample_batch  # Single sample case
            else:
                z_sample_i = z_sample_batch.unsqueeze(0)  # Add batch dimension if needed
        else:
            z_sample_i = node_leaves[i]['z_sample']  # shape [1, latent_dim] for single sample forward

        # Record leaf meta
        leaf_info = {
            'depth': depth_level,
            'node_type': 'leaf',
            'cumulative_prob': prob_i.item(),
            'leaf_id': i,
            'node_id': id(leaf_node)
        }
        leaf_data.append(leaf_info)

        # Recompute reconstruction deterministically from the SAME z_sample used by the forward pass
        dec = leaf_node.decoder
        reconstruction = dec(z_sample_i)
        reconstructions.append(reconstruction)
        leaves_prob.append(prob_i)

    # Use the EXACT same reconstruction loss - from batch if provided, otherwise from this forward
    if batch_outputs is not None and 'rec_loss_samples' in batch_outputs:
        final_rec_loss = batch_outputs['rec_loss_samples'][sample_idx_in_batch].item()
    elif standard_rec_loss is not None:
        final_rec_loss = standard_rec_loss
    else:
        # Fallback
        final_rec_loss = standard_outputs['rec_loss'].item()
    
    # Get leaf assignment (highest probability leaf)
    if len(leaves_prob) > 0:
        leaf_probs = torch.cat([prob.unsqueeze(-1) for prob in leaves_prob], dim=-1)
        leaf_assignment = leaf_probs.argmax(dim=-1).item()
    else:
        leaf_assignment = 0
    
    # Add reconstruction losses to leaf data
    # Compute individual leaf reconstruction losses using the SAME method as the model
    # This ensures: sum(cumulative_prob[i] * leaf_rec_loss[i]) == final_rec_loss
    total_weighted_loss = 0.0
    
    for i, leaf_info in enumerate(leaf_data):
        if i < len(reconstructions):
            # Flatten input and reconstruction for loss computation
            x_flat = torch.flatten(x, start_dim=1)
            recon_flat = torch.flatten(reconstructions[i], start_dim=1)
            
            # Use the same loss function as the model
            if model.activation == "sigmoid":
                # Binary cross entropy loss (matches loss_reconstruction_binary)
                # reduction='none' gives loss per pixel, .sum(dim=-1) sums over all pixels
                leaf_rec_loss = torch.nn.functional.binary_cross_entropy(
                    input=recon_flat, target=x_flat, reduction='none'
                ).sum(dim=-1).item()
            elif model.activation == "mse":
                # MSE loss (matches loss_reconstruction_mse)
                # reduction='none' gives loss per pixel, .sum(dim=-1) sums over all pixels
                leaf_rec_loss = torch.nn.functional.mse_loss(
                    input=recon_flat, target=x_flat, reduction='none'
                ).sum(dim=-1).item()
            else:
                # Fallback
                leaf_rec_loss = torch.nn.functional.mse_loss(x, reconstructions[i]).item()
            
            leaf_info['rec_loss'] = leaf_rec_loss
            
            # Accumulate weighted loss for verification
            total_weighted_loss += leaves_prob[i].item() * leaf_rec_loss
        else:
            leaf_info['rec_loss'] = 0.0
    
    # Return results including the weighted sum so the caller can decide on logging
    return {
        'final_rec_loss': final_rec_loss,
        'leaf_assignment': leaf_assignment,
        'node_data': node_data,
        'leaf_data': leaf_data,
        'weighted_sum': total_weighted_loss
    }


def create_combined_testset_with_fmnist(configs):
    """
    Create combined test set with MNIST + Fashion-MNIST (same as anomaly_experiment_fmnist.py)
    """
    print("Creating combined test set with Fashion-MNIST...")
    
    # Get MNIST test set (normal way)
    mnist_configs = configs.copy()
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
    
    # Fashion-MNIST labels: all 1 (anomaly)
    fmnist_binary_labels = np.ones(len(fmnist_subset), dtype=int)
    
    # Combine binary labels
    combined_binary_labels = np.concatenate([np.zeros(len(mnist_labels)), fmnist_binary_labels])
    
    # Create true labels for reference (MNIST digits + Fashion-MNIST classes)
    # Get actual Fashion-MNIST class labels for the selected samples
    fmnist_true_labels = []
    for idx in fmnist_indices:
        fmnist_true_labels.append(fmnist_testset.targets[idx].item())
    fmnist_true_labels = np.array(fmnist_true_labels)
    
    combined_true_labels = np.concatenate([mnist_labels, fmnist_true_labels])
    
    print(f"Normal samples (MNIST): {len(mnist_labels)}")
    print(f"Anomaly samples (Fashion-MNIST): {len(fmnist_subset)}")
    print(f"Total anomaly samples: {np.sum(combined_binary_labels)}")
    print(f"Anomaly rate: {np.sum(combined_binary_labels) / len(combined_binary_labels) * 100:.1f}%")
    
    return combined_testset, combined_binary_labels, combined_true_labels, mnist_trainset, mnist_trainset_eval


def run_detailed_probability_analysis(configs, device, anomaly_digit=0, test_fraction=1.0):
    """
    Run detailed probability analysis experiment with Fashion-MNIST
    
    Parameters:
    -----------
    test_fraction : float, optional (default=1.0)
        Fraction of test dataset to use. Set to 0.1 for 1/10th of data, 1.0 for full dataset.
        Useful for faster testing/debugging.
    """
    print("="*60)
    print("TREEVAE DETAILED PROBABILITY ANALYSIS WITH FASHION-MNIST")
    print("="*60)
    print(f"Dataset: {configs['data']['data_name']} + Fashion-MNIST")
    print(f"Anomaly Digit: {anomaly_digit}")
    print(f"Device: {device}")
    if test_fraction < 1.0:
        print(f"⚠️  TEST MODE: Using {test_fraction*100:.1f}% of test data ({test_fraction} fraction)")
    print("="*60)
    
    # Set anomaly digit in config
    configs['data']['anomaly_digit'] = anomaly_digit
    
    # Create combined test set with Fashion-MNIST
    combined_testset, anomaly_labels_binary, true_labels, trainset, trainset_eval = create_combined_testset_with_fmnist(configs)
    
    # Subsample test dataset if test_fraction < 1.0
    original_test_size = len(combined_testset)
    if test_fraction < 1.0:
        np.random.seed(configs['globals']['seed'])
        test_size = int(len(combined_testset) * test_fraction)
        test_indices = np.random.choice(len(combined_testset), test_size, replace=False)
        test_indices = np.sort(test_indices)  # Keep original order
        
        combined_testset = Subset(combined_testset, test_indices)
        anomaly_labels_binary = anomaly_labels_binary[test_indices]
        true_labels = np.array(true_labels)[test_indices] if not isinstance(true_labels, np.ndarray) else true_labels[test_indices]
        true_labels = true_labels.tolist() if isinstance(true_labels, np.ndarray) else true_labels
        
        print(f"Subsampled test dataset: {original_test_size} → {len(combined_testset)} samples")
    
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(combined_testset)}")
    
    # Create experiment path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = Path(f"models/experiments/{configs['data']['data_name']}_detailed_analysis_{timestamp}")
    print(f"Experiment path: {experiment_path}")
    experiment_path.mkdir(parents=True, exist_ok=True)
    configs['experiment_path'] = experiment_path
    
    # Initialize wandb
    import wandb
    import uuid
    import time
    
    project_dir = Path(__file__).absolute().parent
    timestr = time.strftime("%Y%m%d-%H%M%S")
    ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
    experiment_path = configs['globals']['results_dir'] / configs['data']['data_name'] / ex_name
    experiment_path.mkdir(parents=True)
    os.makedirs(os.path.join(project_dir, '../models/logs', ex_name))
    print("Experiment path: ", experiment_path)
    
    # Wandb setup
    os.environ['WANDB_CACHE_DIR'] = os.path.join(project_dir, '../wandb', '.cache', 'wandb')
    os.environ["WANDB_SILENT"] = "true"
    
    wandb.init(
        project="treevae-detailed-analysis",
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
    
    # Reproducibility
    from utils.utils import reset_random_seeds
    reset_random_seeds(configs['globals']['seed'])
    
    # Train the model
    print("\n" + "="*60)
    print("STARTING TREEVAE TRAINING")
    print("="*60)
    
    model = run_tree(trainset, trainset_eval, combined_testset, device, configs)
    
    print("\n" + "="*60)
    print("TREEVAE TRAINING COMPLETED - STARTING DETAILED ANALYSIS")
    print("="*60)
    
    # Use existing validation function for comprehensive evaluation - EXACT same as anomaly_experiment_fmnist.py
    val_tree(trainset, combined_testset, model, device, experiment_path, configs)
    
    # Get detailed probabilities for analysis
    print("\nCalculating detailed probabilities for all test samples...")
    
    # Create a data loader for the test set
    from utils.data_utils import get_gen
    test_loader = get_gen(combined_testset, configs, validation=True, shuffle=False)
    
    # Get detailed probabilities
    sample_data, node_data, leaf_data = get_detailed_probabilities_from_loader(model, test_loader, device, anomaly_labels_binary, true_labels)
    
    # Create DataFrames
    sample_df = pd.DataFrame(sample_data)
    node_df = pd.DataFrame(node_data)
    leaf_df = pd.DataFrame(leaf_data)
    
    # Create unique output directory
    unique_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    seed = configs['globals']['seed']
    
    output_dir = f"eval_datasets/detailed_analysis_digit_{anomaly_digit}_seed_{seed}_{unique_timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save only two CSV files: sample summary and leaf reconstructions
    sample_df.to_csv(os.path.join(output_dir, f"sample_summary_digit_{anomaly_digit}_seed_{seed}_{unique_timestamp}.csv"), index=False)
    leaf_df.to_csv(os.path.join(output_dir, f"leaf_reconstructions_digit_{anomaly_digit}_seed_{seed}_{unique_timestamp}.csv"), index=False)
    print(f"Detailed analysis saved CSVs in: {output_dir}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DETAILED PROBABILITY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Dataset: {configs['data']['data_name']}")
    print(f"Seed: {seed}")
    print(f"Total samples: {len(sample_df)}")
    print(f"Total leaves: {len(leaf_df)}")
    print(f"Average final rec loss: {sample_df['final_rec_loss'].mean():.4f}")
    print(f"Std final rec loss: {sample_df['final_rec_loss'].std():.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    
    # Show sample results
    print("\nSample-level data (first 5 rows):")
    print(sample_df.head())
    
    print("\nLeaf-level data (first 5 rows):")
    print(leaf_df.head())
    
    # Finish wandb
    wandb.finish(quiet=True)
    
    return sample_df, leaf_df, output_dir


def main():
    """
    Main function for detailed probability analysis
    """
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)

    parser = argparse.ArgumentParser(description='TreeVAE Detailed Probability Analysis')

    # Reuse existing arguments from main.py
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
    
    # Test dataset subsampling (for faster testing/debugging)
    parser.add_argument('--test_fraction', type=float, default=1.0,
                        help='Fraction of test dataset to use (0.0-1.0). Default 1.0 (full dataset). Use 0.1 for 1/10th of data.')

    args = parser.parse_args()
    configs = prepare_config(args, project_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print experiment information
    print("="*60)
    print("🔬 TREEVAE DETAILED PROBABILITY ANALYSIS WITH FASHION-MNIST")
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
        sample_df, leaf_df, output_dir = run_detailed_probability_analysis(configs, device, args.anomaly_digit, args.test_fraction)
        
        print(f"\n🎉 Detailed analysis completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📄 Files: detailed_probability_analysis_digit_{args.anomaly_digit}_seed_{configs['globals']['seed']}_*")
        print(f"📊 Content: Sample_Summary, Node_Probabilities, Leaf_Reconstructions, Summary_Statistics")
        
    except Exception as e:
        print(f"❌ Error running detailed analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
