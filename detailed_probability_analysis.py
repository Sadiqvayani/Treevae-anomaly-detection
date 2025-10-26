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
    Get detailed routing probabilities and reconstruction losses for all samples
    """
    model.eval()
    
    # Lists to store all data
    all_sample_data = []
    all_node_data = []
    all_leaf_data = []
    
    with torch.no_grad():
        for batch_idx, (batch_data, batch_labels) in enumerate(tqdm(data_loader, desc="Processing batches")):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Process each sample in the batch
            for sample_idx in range(batch_data.size(0)):
                sample_id = batch_idx * data_loader.batch_size + sample_idx
                single_sample = batch_data[sample_idx:sample_idx+1]  # Keep batch dimension
                single_label = batch_labels[sample_idx]
                
                # Get detailed outputs for this sample
                sample_outputs = get_single_sample_detailed_outputs(model, single_sample, device)
                
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
                
                # Store sample-level data
                sample_data = {
                    'sample_id': sample_id,
                    'true_digit': true_digit,
                    'is_anomaly': is_anomaly,
                    'final_rec_loss': sample_outputs['final_rec_loss'],
                    'leaf_assignment': sample_outputs['leaf_assignment'],
                    'dataset_source': dataset_source,
                    'class_label': class_label,
                    'num_nodes_processed': len(sample_outputs['node_data']),
                    'num_leaves': len(sample_outputs['leaf_data'])
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


def get_single_sample_detailed_outputs(model, x, device):
    """
    Get detailed outputs for a single sample including all routing probabilities
    """
    epsilon = 1e-7
    
    # Compute deterministic bottom up
    d = x
    encoders = []
    emb_contr = []

    for i in range(0, len(model.hidden_layers)):
        d, _, _ = model.bottom_up[i](d)
        encoders.append(d)
        
        # Pass through contrastive MLP's if contrastive learning is selected
        if hasattr(model, 'augmentation_method') and 'instancewise_full' in model.augmentation_method:
            _, emb_c, _ = model.contrastive_mlp[i](d)
            emb_contr.append(emb_c)
        elif hasattr(model, 'augmentation_method') and 'instancewise_first' in model.augmentation_method:
            if i == 0:
                _, emb_c, _ = model.contrastive_mlp[i](d)
                emb_contr.append(emb_c)

    # Create a list of nodes of the tree that need to be processed
    list_nodes = [{'node': model.tree, 'depth': 0, 'prob': torch.ones(x.size(0), device=device), 'z_parent_sample': None}]
    
    # Initialize tracking lists
    node_data = []
    leaf_data = []
    leaves_prob = []
    reconstructions = []
    
    # Process all nodes
    while len(list_nodes) != 0:
        current_node = list_nodes.pop(0)
        node, depth_level, prob = current_node['node'], current_node['depth'], current_node['prob']
        z_parent_sample = current_node['z_parent_sample']
        
        # Access deterministic bottom up mu and sigma hat
        d = encoders[-(1+depth_level)]
        z_mu_q_hat, z_sigma_q_hat = node.dense(d)

        # Handle root vs other nodes
        if depth_level == 0:
            z_mu_p, z_sigma_p = torch.zeros_like(z_mu_q_hat, device=device), torch.ones_like(z_sigma_q_hat, device=device)
            z_mu_q, z_sigma_q = z_mu_q_hat, z_sigma_q_hat
        else:
            _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample)
            from utils.model_utils import compute_posterior
            z_mu_q, z_sigma_q = compute_posterior(z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p)

        # Compute sample z
        z = torch.distributions.Independent(torch.distributions.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 1)
        z_sample = z.rsample()

        # If there is a router (internal node)
        if node.router is not None:
            # Compute routing probabilities
            prob_child_left = node.router(z_sample).squeeze()
            prob_child_left_q = node.routers_q(d).squeeze()
            
            # Store node data
            node_info = {
                'depth': depth_level,
                'node_type': 'internal',
                'cumulative_prob': prob.item(),
                'routing_prob_left': prob_child_left_q.item(),
                'routing_prob_right': (1 - prob_child_left_q).item(),
                'top_down_prob_left': prob_child_left.item(),
                'top_down_prob_right': (1 - prob_child_left).item(),
                'node_id': id(node)
            }
            node_data.append(node_info)
            
            # Add children to processing queue
            prob_node_left, prob_node_right = prob * prob_child_left_q, prob * (1 - prob_child_left_q)
            node_left, node_right = node.left, node.right
            list_nodes.append(
                {'node': node_left, 'depth': depth_level + 1, 'prob': prob_node_left, 'z_parent_sample': z_sample})
            list_nodes.append({'node': node_right, 'depth': depth_level + 1, 'prob': prob_node_right,
                            'z_parent_sample': z_sample})

        # If there is a decoder (leaf node)
        elif node.decoder is not None:
            # Store leaf data
            leaf_info = {
                'depth': depth_level,
                'node_type': 'leaf',
                'cumulative_prob': prob.item(),
                'leaf_id': len(leaves_prob),  # Leaf index
                'node_id': id(node)
            }
            leaf_data.append(leaf_info)
            
            # Compute reconstruction
            dec = node.decoder
            reconstruction = dec(z_sample)
            reconstructions.append(reconstruction)
            leaves_prob.append(prob)

        # Handle pruned nodes (internal nodes with only one child)
        elif node.router is None and node.decoder is None:
            node_left, node_right = node.left, node.right
            child = node_left if node_left is not None else node_right
            list_nodes.append(
                {'node': child, 'depth': depth_level + 1, 'prob': prob, 'z_parent_sample': z_sample})

    # Compute final reconstruction loss
    if len(reconstructions) > 0:
        # Use the model's loss function
        if hasattr(model, 'loss'):
            rec_losses = model.loss(x, reconstructions, leaves_prob)
            final_rec_loss = rec_losses.item()
        else:
            # Fallback: compute MSE loss
            final_rec_loss = torch.nn.functional.mse_loss(x, reconstructions[0]).item()
    else:
        final_rec_loss = 0.0
    
    # Get leaf assignment (highest probability leaf)
    if len(leaves_prob) > 0:
        leaf_probs = torch.cat([prob.unsqueeze(-1) for prob in leaves_prob], dim=-1)
        leaf_assignment = leaf_probs.argmax(dim=-1).item()
    else:
        leaf_assignment = 0
    
    # Add reconstruction losses to leaf data
    for i, leaf_info in enumerate(leaf_data):
        if i < len(reconstructions):
            # Compute individual leaf reconstruction loss
            leaf_rec_loss = torch.nn.functional.mse_loss(x, reconstructions[i]).item()
            leaf_info['rec_loss'] = leaf_rec_loss
        else:
            leaf_info['rec_loss'] = 0.0
    
    return {
        'final_rec_loss': final_rec_loss,
        'leaf_assignment': leaf_assignment,
        'node_data': node_data,
        'leaf_data': leaf_data
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


def run_detailed_probability_analysis(configs, device):
    """
    Run detailed probability analysis experiment with Fashion-MNIST
    """
    print("="*60)
    print("TREEVAE DETAILED PROBABILITY ANALYSIS WITH FASHION-MNIST")
    print("="*60)
    print(f"Dataset: {configs['data']['data_name']} + Fashion-MNIST")
    print(f"Device: {device}")
    print("="*60)
    
    # Create combined test set with Fashion-MNIST
    combined_testset, anomaly_labels_binary, true_labels, trainset, trainset_eval = create_combined_testset_with_fmnist(configs)
    
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
    
    model = run_tree(trainset, trainset_eval, testset, device, configs)
    
    print("\n" + "="*60)
    print("TREEVAE TRAINING COMPLETED - STARTING DETAILED ANALYSIS")
    print("="*60)
    
    # Get detailed probabilities
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
    
    output_dir = f"eval_datasets/detailed_analysis_seed_{seed}_{unique_timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to Excel with multiple sheets
    excel_file = os.path.join(output_dir, f"detailed_probability_analysis_seed_{seed}_{unique_timestamp}.xlsx")
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Sample-level data
        sample_df.to_excel(writer, sheet_name='Sample_Summary', index=False)
        
        # Node-level data (all routing probabilities)
        node_df.to_excel(writer, sheet_name='Node_Probabilities', index=False)
        
        # Leaf-level data (reconstruction losses)
        leaf_df.to_excel(writer, sheet_name='Leaf_Reconstructions', index=False)
        
        # Create summary statistics
        summary_stats = {
            'Metric': [
                'Total Samples',
                'Total Nodes',
                'Total Leaves',
                'Max Depth',
                'Avg Final Rec Loss',
                'Std Final Rec Loss',
                'Min Final Rec Loss',
                'Max Final Rec Loss'
            ],
            'Value': [
                len(sample_df),
                len(node_df),
                len(leaf_df),
                max(node_df['depth'].max(), leaf_df['depth'].max()) if len(node_df) > 0 or len(leaf_df) > 0 else 0,
                sample_df['final_rec_loss'].mean(),
                sample_df['final_rec_loss'].std(),
                sample_df['final_rec_loss'].min(),
                sample_df['final_rec_loss'].max()
            ]
        }
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    print(f"Detailed analysis saved to: {excel_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DETAILED PROBABILITY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Dataset: {configs['data']['data_name']}")
    print(f"Seed: {seed}")
    print(f"Total samples: {len(sample_df)}")
    print(f"Total nodes: {len(node_df)}")
    print(f"Total leaves: {len(leaf_df)}")
    print(f"Max depth: {max(node_df['depth'].max(), leaf_df['depth'].max()) if len(node_df) > 0 or len(leaf_df) > 0 else 0}")
    print(f"Average final rec loss: {sample_df['final_rec_loss'].mean():.4f}")
    print(f"Std final rec loss: {sample_df['final_rec_loss'].std():.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    
    # Show sample results
    print("\nSample-level data (first 5 rows):")
    print(sample_df.head())
    
    print("\nNode-level data (first 10 rows):")
    print(node_df.head(10))
    
    print("\nLeaf-level data (first 5 rows):")
    print(leaf_df.head())
    
    # Finish wandb
    wandb.finish(quiet=True)
    
    return sample_df, node_df, leaf_df, output_dir


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

    args = parser.parse_args()
    configs = prepare_config(args, project_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print experiment information
    print("="*60)
    print("🔬 TREEVAE DETAILED PROBABILITY ANALYSIS")
    print("="*60)
    print(f"📊 Dataset: {configs['data']['data_name']}")
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
        sample_df, node_df, leaf_df, output_dir = run_detailed_probability_analysis(configs, device)
        
        print(f"\n🎉 Detailed analysis completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📄 Excel file: detailed_probability_analysis_seed_{configs['globals']['seed']}_*.xlsx")
        print(f"📊 Sheets: Sample_Summary, Node_Probabilities, Leaf_Reconstructions, Summary_Statistics")
        
    except Exception as e:
        print(f"❌ Error running detailed analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
