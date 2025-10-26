#!/usr/bin/env python3
"""
Test Trained TreeVAE Model - Testing Only Script
- Loads any trained TreeVAE model
- Dynamically detects anomaly digit from config
- Tests on different anomaly scenarios
- Generates AUC, CSV outputs, and ROC curves
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
import yaml
import torchvision
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader, Subset, ConcatDataset

# Import existing functions from the project
from utils.data_utils import get_data, get_gen
from models.model import TreeVAE
from train.validate_tree import val_tree


def load_trained_model(experiment_path):
    """
    Load a trained TreeVAE model and its configuration using the proper reconstruction method
    
    Parameters
    ----------
    experiment_path : str or Path
        Path to the experiment directory containing model_weights.pt, config.yaml, and data_tree.npy
    
    Returns
    -------
    model : TreeVAE
        Loaded trained model
    config : dict
        Configuration dictionary
    """
    experiment_path = Path(experiment_path)
    
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment path does not exist: {experiment_path}")
    
    # Load config with custom loader to handle pathlib.PosixPath
    def pathlib_constructor(loader, node):
        value = loader.construct_sequence(node)
        return Path(*value)
    
    # Create a custom loader class
    class CustomLoader(yaml.Loader):
        pass
    
    CustomLoader.add_constructor('tag:yaml.org,2002:python/object/apply:pathlib.PosixPath', pathlib_constructor)
    
    config_file = experiment_path / 'config.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=CustomLoader)
    
    # Load model weights
    model_file = experiment_path / 'model_weights.pt'
    if not model_file.exists():
        raise FileNotFoundError(f"Model weights file not found: {model_file}")
    
    # Create model with same architecture
    model = TreeVAE(**config['training'])
    
    # Load trained weights with strict=False to handle architecture mismatches
    # This allows loading models with different tree structures
    state_dict = torch.load(model_file, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys (will use random initialization): {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys (will be ignored): {len(unexpected_keys)} keys")
    
    model.eval()
    
    print(f"✅ Loaded trained model from: {experiment_path}")
    print(f"📊 Model trained on: {config['data']['data_name']}")
    print(f"🌱 Seed: {config['globals']['seed']}")
    print(f"🌳 Tree depth: {model.compute_depth()}")
    print(f"🍃 Number of leaves: {len(model.compute_leaves())}")
    
    return model, config


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
    
    return combined_testset, combined_binary_labels, combined_true_labels


def test_model_on_scenario(model, config, test_scenario, device):
    """
    Test trained model on a specific scenario
    
    Parameters
    ----------
    model : TreeVAE
        Trained model
    config : dict
        Model configuration
    test_scenario : dict
        Test scenario configuration
    device : torch.device
        Device to run on
    
    Returns
    -------
    results_df : pd.DataFrame
        Results dataframe
    auc_score : float
        AUC score
    fpr, tpr : np.array
        ROC curve data
    output_dir : str
        Output directory path
    """
    print(f"\n{'='*60}")
    print(f"TESTING SCENARIO: {test_scenario['name']}")
    print(f"{'='*60}")
    print(f"Anomaly digit: {test_scenario['anomaly_digit']}")
    print(f"Test type: {test_scenario['test_type']}")
    print(f"{'='*60}")
    
    # Create test config for this scenario
    test_config = config.copy()
    test_config['data']['anomaly_digit'] = test_scenario['anomaly_digit']
    
    if test_scenario['test_type'] == 'mnist_only':
        # Test on MNIST only
        _, _, testset = get_data(test_config)
        
        # Create binary labels for anomaly detection
        test_labels = []
        for _, labels in testset:
            test_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else [labels])
        test_labels = np.array(test_labels)
        anomaly_labels_binary = (test_labels == test_scenario['anomaly_digit']).astype(int)
        
        true_labels = test_labels
        dataset_sources = ['MNIST'] * len(testset)
        class_labels = [f"MNIST_{label}" for label in test_labels]
        
    elif test_scenario['test_type'] == 'mnist_fmnist':
        # Test on MNIST + Fashion-MNIST
        testset, anomaly_labels_binary, true_labels = create_combined_testset_with_fmnist(test_config, test_scenario['anomaly_digit'])
        
        # Create dataset source and class labels
        dataset_sources = ['MNIST'] * (len(testset) - 1000) + ['Fashion-MNIST'] * 1000
        class_labels = []
        
        # MNIST class labels (0-9)
        for i in range(len(testset) - 1000):
            class_labels.append(f"MNIST_{true_labels[i]}")
        
        # Fashion-MNIST class labels (0-9)
        for i in range(len(testset) - 1000, len(testset)):
            class_labels.append(f"FMNIST_{true_labels[i]}")
    
    else:
        raise ValueError(f"Unknown test type: {test_scenario['test_type']}")
    
    # Create data loader
    test_loader = get_gen(testset, test_config, validation=True, shuffle=False)
    
    # First run comprehensive validation (like in training)
    print("\nRunning comprehensive model evaluation...")
    
    # Create a temporary experiment path for validation
    temp_experiment_path = Path(f"temp_eval_{test_scenario['name']}")
    temp_experiment_path.mkdir(exist_ok=True)
    
    # Run validation to get all metrics (accuracy, NMI, ARI, etc.)
    val_tree(testset, testset, model, device, temp_experiment_path, test_config)
    
    # Get reconstruction losses
    reconstruction_losses, leaf_assignments = get_reconstruction_losses_from_loader(model, test_loader, device)
    
    # Calculate ROC curve and AUC
    auc_score = roc_auc_score(anomaly_labels_binary, reconstruction_losses)
    fpr, tpr, thresholds = roc_curve(anomaly_labels_binary, reconstruction_losses)
    
    print(f"\nAnomaly Detection AUC: {auc_score:.4f}")
    
    # Print detailed reconstruction loss analysis (like in training)
    print(f"\nReconstruction Loss Analysis:")
    normal_losses = reconstruction_losses[anomaly_labels_binary == 0]
    anomaly_losses = reconstruction_losses[anomaly_labels_binary == 1]
    
    print(f"Normal samples (digits {[d for d in range(10) if d != test_scenario['anomaly_digit']]}):")
    print(f"  Count: {len(normal_losses)}")
    print(f"  Mean reconstruction loss: {normal_losses.mean():.4f}")
    print(f"  Std reconstruction loss: {normal_losses.std():.4f}")
    print(f"  Min reconstruction loss: {normal_losses.min():.4f}")
    print(f"  Max reconstruction loss: {normal_losses.max():.4f}")
    
    print(f"\nAnomaly samples (digit {test_scenario['anomaly_digit']}):")
    print(f"  Count: {len(anomaly_losses)}")
    print(f"  Mean reconstruction loss: {anomaly_losses.mean():.4f}")
    print(f"  Std reconstruction loss: {anomaly_losses.std():.4f}")
    print(f"  Min reconstruction loss: {anomaly_losses.min():.4f}")
    print(f"  Max reconstruction loss: {anomaly_losses.max():.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'sample_id': range(len(testset)),
        'true_digit': true_labels,
        'is_anomaly': anomaly_labels_binary.astype(int),
        'reconstruction_loss': reconstruction_losses,
        'leaf_assignment': leaf_assignments,
        'dataset_source': dataset_sources,
        'class_label': class_labels
    })
    
    # Create unique output directory for results
    unique_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    seed = config['globals']['seed']
    
    output_dir = f"test_results/{test_scenario['name']}_seed_{seed}_{unique_timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results CSV
    results_file = os.path.join(output_dir, f"{test_scenario['name']}_seed_{seed}_{unique_timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Results CSV saved to: {results_file}")
    
    # Save results Excel
    try:
        excel_file = os.path.join(output_dir, f"{test_scenario['name']}_seed_{seed}_{unique_timestamp}.xlsx")
        results_df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"Results Excel saved to: {excel_file}")
    except ImportError:
        print("⚠️  Excel export skipped: openpyxl not available")
    
    # Plot and save ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {test_scenario["name"]}\n(Anomaly Digit: {test_scenario["anomaly_digit"]}, Seed: {seed})', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    roc_file = os.path.join(output_dir, f"roc_curve_{test_scenario['name']}_seed_{seed}_{unique_timestamp}.png")
    plt.savefig(roc_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {roc_file}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"TEST RESULTS SUMMARY: {test_scenario['name']}")
    print(f"{'='*60}")
    print(f"Test type: {test_scenario['test_type']}")
    print(f"Anomaly digit: {test_scenario['anomaly_digit']}")
    print(f"Seed: {seed}")
    print(f"Total test samples: {len(testset)}")
    print(f"Normal samples: {np.sum(anomaly_labels_binary == 0)}")
    print(f"Anomaly samples: {np.sum(anomaly_labels_binary == 1)}")
    print(f"Anomaly rate: {np.sum(anomaly_labels_binary) / len(anomaly_labels_binary) * 100:.1f}%")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    # Print sample results (like in training)
    print(f"\nSample results (first 10 rows):")
    print(results_df[['sample_id', 'true_digit', 'is_anomaly', 'reconstruction_loss', 'dataset_source', 'class_label']].head(10).to_string(index=False))
    
    print(f"\n[{len(results_df)} rows x {len(results_df.columns)} columns]")
    
    return results_df, auc_score, fpr, tpr, output_dir


def main():
    """
    Main function for testing trained TreeVAE models
    """
    parser = argparse.ArgumentParser(description='Test Trained TreeVAE Model')
    
    # Model path argument
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model directory (containing model_weights.pt and config.yaml)')
    
    # Test scenarios
    parser.add_argument('--test_anomaly_digits', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help='List of anomaly digits to test (0-9)')
    
    parser.add_argument('--test_type', type=str, choices=['mnist_only', 'mnist_fmnist', 'both'], default='both',
                        help='Type of test to run: mnist_only, mnist_fmnist, or both')
    
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed to use (overrides model config)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("🔬 TREEVAE MODEL TESTING SCRIPT")
    print("="*60)
    print(f"📁 Model path: {args.model_path}")
    print(f"🎯 Test anomaly digits: {args.test_anomaly_digits}")
    print(f"🧪 Test type: {args.test_type}")
    print(f"💻 Device: {device}")
    print("="*60)
    
    try:
        # Load trained model
        model, config = load_trained_model(args.model_path)
        model.to(device)
        
        # Override seed if specified
        if args.seed is not None:
            config['globals']['seed'] = args.seed
            print(f"🌱 Overriding seed to: {args.seed}")
        
        # Detect original anomaly digit from config
        original_anomaly_digit = config['data'].get('anomaly_digit', 0)
        print(f"🔍 Detected original anomaly digit from config: {original_anomaly_digit}")
        
        # Create test scenarios
        test_scenarios = []
        
        for anomaly_digit in args.test_anomaly_digits:
            if args.test_type in ['mnist_only', 'both']:
                test_scenarios.append({
                    'name': f'MNIST_Anomaly_{anomaly_digit}',
                    'anomaly_digit': anomaly_digit,
                    'test_type': 'mnist_only'
                })
            
            if args.test_type in ['mnist_fmnist', 'both']:
                test_scenarios.append({
                    'name': f'MNIST_FMNIST_Anomaly_{anomaly_digit}',
                    'anomaly_digit': anomaly_digit,
                    'test_type': 'mnist_fmnist'
                })
        
        print(f"\n📋 Created {len(test_scenarios)} test scenarios")
        
        # Run tests
        all_results = {}
        
        for scenario in test_scenarios:
            try:
                results_df, auc_score, fpr, tpr, output_dir = test_model_on_scenario(
                    model, config, scenario, device
                )
                
                all_results[scenario['name']] = {
                    'auc_score': auc_score,
                    'output_dir': output_dir,
                    'results_df': results_df
                }
                
            except Exception as e:
                print(f"❌ Error testing scenario {scenario['name']}: {e}")
                import traceback
                traceback.print_exc()
        
        # Print final summary
        print(f"\n{'='*60}")
        print("🎉 ALL TESTS COMPLETED")
        print(f"{'='*60}")
        
        for scenario_name, result in all_results.items():
            print(f"📊 {scenario_name}: AUC = {result['auc_score']:.4f}")
            print(f"📁 Results: {result['output_dir']}")
            
            # Show sample of reconstruction losses for the first result
            if len(all_results) == 1:  # Only show details if testing single anomaly
                df = result['results_df']
                print(f"\n📋 Sample Results for {scenario_name}:")
                print(f"Total samples: {len(df)}")
                print(f"Normal samples: {len(df[df['is_anomaly'] == 0])}")
                print(f"Anomaly samples: {len(df[df['is_anomaly'] == 1])}")
                print(f"\nReconstruction Loss Statistics:")
                print(f"  Normal samples - Mean: {df[df['is_anomaly'] == 0]['reconstruction_loss'].mean():.4f}")
                print(f"  Normal samples - Std:  {df[df['is_anomaly'] == 0]['reconstruction_loss'].std():.4f}")
                print(f"  Anomaly samples - Mean: {df[df['is_anomaly'] == 1]['reconstruction_loss'].mean():.4f}")
                print(f"  Anomaly samples - Std:  {df[df['is_anomaly'] == 1]['reconstruction_loss'].std():.4f}")
                print(f"\nFirst 10 samples:")
                print(df[['sample_id', 'true_digit', 'is_anomaly', 'reconstruction_loss']].head(10).to_string(index=False))
        
        print(f"\n✅ Successfully tested {len(all_results)} scenarios")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
