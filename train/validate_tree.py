import wandb
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import gc
import yaml
import torch
import scipy
from tqdm import tqdm

from utils.data_utils import get_gen
from utils.utils import cluster_acc, dendrogram_purity, leaf_purity
from utils.training_utils import compute_leaves, validate_one_epoch, Custom_Metrics, predict
from utils.model_utils import construct_data_tree
from models.losses import loss_reconstruction_cov_mse_eval


def val_tree(trainset, testset, model, device, experiment_path, configs):
    """
    Run the validation of a trained instance of TreeVAE on both the train and test datasets. All final results and
    validations will be stored in Wandb, while the most important ones will be also printed out in the terminal.

    Parameters
    ----------
    trainset: torch.utils.data.Dataset
        The train dataset
    testset: torch.utils.data.Dataset
        The test dataset
    model: models.model.TreeVAE
        The trained TreeVAE model
    device: torch.device
        The device in which to validate the model
    experiment_path: str
        The experimental path where to store the tree
    configs: dict
        The config setting for training and validating TreeVAE defined in configs or in the command line
    """

    ############ Training set performance ############

    # get the data loader 
    
    gen_train_eval = get_gen(trainset, configs, validation=True, shuffle=False)
    y_train = trainset.dataset.targets[trainset.indices].numpy()


    # compute the leaf probabilities
    prob_leaves_train = predict(gen_train_eval, model, device, 'prob_leaves')
    _ = gc.collect()
    # compute the predicted cluster
    y_train_pred = np.squeeze(np.argmax(prob_leaves_train, axis=-1)).numpy()
    # compute clustering metrics
    acc, idx = cluster_acc(y_train, y_train_pred, return_index=True)
    nmi = normalized_mutual_info_score(y_train, y_train_pred)
    ari = adjusted_rand_score(y_train, y_train_pred)
    wandb.log({"Train Accuracy": acc, "Train Normalized Mutual Information": nmi, "Train Adjusted Rand Index": ari})
    # compute confusion matrix
    swap = dict(zip(range(len(idx)), idx))
    y_wandb = np.array([swap[i] for i in y_train_pred], dtype=np.uint8)
    wandb.log({"Train_confusion_matrix":
                   wandb.plot.confusion_matrix(probs=None, y_true=y_train, preds=y_wandb, class_names=range(len(idx)))})

    ############ Test set performance ############

    # get the data loader 
    gen_test = get_gen(testset, configs, validation=True, shuffle=False)
    y_test = testset.dataset.targets[testset.indices].numpy()
    # compute one validation pass through the test set to log losses
    metrics_calc_test = Custom_Metrics(device)
    validate_one_epoch(gen_test, model, metrics_calc_test, 0, device, test=True)
    _ = gc.collect()
    # predict the leaf probabilities and the leaves
    node_leaves_test, prob_leaves_test = predict(gen_test, model, device, 'node_leaves', 'prob_leaves')
    _ = gc.collect()


    # compute the predicted cluster
    y_test_pred = np.squeeze(np.argmax(prob_leaves_test, axis=-1)).numpy()
    # Calculate clustering metrics
    acc, idx = cluster_acc(y_test, y_test_pred, return_index=True)
    nmi = normalized_mutual_info_score(y_test, y_test_pred)
    ari = adjusted_rand_score(y_test, y_test_pred)
    wandb.log({"Test Accuracy": acc, "Test Normalized Mutual Information": nmi, "Test Adjusted Rand Index": ari})
    # Calculate confusion matrix
    swap = dict(zip(range(len(idx)), idx))
    y_wandb = np.array([swap[i] for i in y_test_pred], dtype=np.uint8)
    wandb.log({"Test_confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                                    y_true=y_test, preds=y_wandb,
                                                                    class_names=range(len(idx)))})

    # Determine indices of samples that fall into each leaf for Dendogram Purity & Leaf Purity
    leaves = compute_leaves(model.tree)
    ind_samples_of_leaves = []
    for i in range(len(leaves)):
        ind_samples_of_leaves.append([leaves[i]['node'], np.where(y_test_pred == i)[0]])
    # Calculate leaf and dedrogram purity
    dp = dendrogram_purity(model.tree, y_test, ind_samples_of_leaves)
    lp = leaf_purity(model.tree, y_test, ind_samples_of_leaves)
    # Note: Only comparable DP & LP values wrt baselines if they have the same n_leaves for all methods
    wandb.log({"Test Dendrogram Purity": dp, "Test Leaf Purity": lp})

    # Save the tree structure of TreeVAE and log it
    data_tree = construct_data_tree(model, y_predicted=y_test_pred, y_true=y_test, n_leaves=len(node_leaves_test),
                                    data_name=configs['data']['data_name'])

    if configs['globals']['save_model']:
        with open(experiment_path / 'data_tree.npy', 'wb') as save_file:
            np.save(save_file, data_tree)
        with open(experiment_path / 'config.yaml', 'w', encoding='utf8') as outfile:
            yaml.dump(configs, outfile, default_flow_style=False, allow_unicode=True)

    table = wandb.Table(columns=["node_id", "node_name", "parent", "size"], data=data_tree)
    fields = {"node_name": "node_name", "node_id": "node_id", "parent": "parent", "size": "size"}
    dendro = wandb.plot_table(vega_spec_name="stacey/flat_tree", data_table=table, fields=fields)
    wandb.log({"dendogram_final": dendro})

    # Printing important results
    print(np.unique(y_test_pred, return_counts=True))
    print("Accuracy:", acc)
    print("Normalized Mutual Information:", nmi)
    print("Adjusted Rand Index:", ari)
    print("Dendrogram Purity:", dp)
    print("Leaf Purity:", lp)
    print("Digits", np.unique(y_test))


 ############ PER-SAMPLE LOSS EXTRACTION ############
    print("\n" + "="*50)
    print("EXTRACTING PER-SAMPLE RECONSTRUCTION LOSSES")
    print("="*50)
    
    # Create a NEW data loader
    gen_test_new = get_gen(testset, configs, validation=True, shuffle=False)
    y_test = testset.dataset.targets[testset.indices].numpy()
    total_samples = len(testset)
    
    print(f"✓ Total test samples: {total_samples}")
    print(f"✓ Batch size: {configs['training']['batch_size']}")
    print(f"✓ Expected batches: {total_samples // configs['training']['batch_size']}")
    
    # Use our simple reliable function
    per_sample_losses = get_all_per_sample_losses_simple(model, gen_test_new, device)
    
    print(f"✓ Gathered {len(per_sample_losses)} per-sample reconstruction losses")
    
    if len(per_sample_losses) != total_samples:
        print(f"❌ WARNING: Expected {total_samples}, got {len(per_sample_losses)}")
        print("❌ There might be an issue with the data loader")
    else:
        print(f"✓ Successfully gathered all {total_samples} samples!")
        print(f"✓ Mean rec_loss: {per_sample_losses.mean():.3f} (should match 113.993)")
        print(f"✓ Min: {per_sample_losses.min():.3f}, Max: {per_sample_losses.max():.3f}")
    
    # Save detailed results
    save_detailed_results(per_sample_losses, y_test, experiment_path)

    # Compute the log-likehood of the test data

    # Compute the log-likehood of the test data
    # ATTENTION it might take a while! If not interested disable the setting in configs
    if configs['training']['compute_ll']:
        compute_likelihood(testset, model, device, configs)
    return

def save_detailed_results(rec_losses, labels, output_path):
    """
    Save detailed reconstruction losses for all test samples
    """
    import csv
    import os
    
    filename = os.path.join(output_path, "detailed_test_results.csv")
    
    with open(filename, 'w', newline='') as csvfile:    # noqa: P101
        writer = csv.writer(csvfile)
        writer.writerow(['Sample_ID', 'Reconstruction_Loss', 'True_Label'])
        
        for i, (rec_loss, label) in enumerate(zip(rec_losses, labels)):
            writer.writerow([i, rec_loss, label])
    
    print(f"✓ Saved detailed results to: {filename}")
    print(f"✓ Samples in file: {len(rec_losses)}")
    
    # Also save a summary
    summary_filename = os.path.join(output_path, "loss_summary.txt")
    with open(summary_filename, 'w') as f:
        f.write(f"Total samples: {len(rec_losses)}\n")
        f.write(f"Mean reconstruction loss: {rec_losses.mean():.6f}\n")
        f.write(f"Min reconstruction loss: {rec_losses.min():.6f}\n")
        f.write(f"Max reconstruction loss: {rec_losses.max():.6f}\n")
        f.write(f"Std reconstruction loss: {rec_losses.std():.6f}\n")
    
    print(f"✓ Saved summary to: {summary_filename}")

def compute_likelihood(testset, model, device, configs):
    """
    Compute the approximated log-likelihood calculated using 1000 importance-weighted samples.

    Parameters
    ----------
    testset: torch.utils.data.Dataset
        The test dataset
    model: models.model.TreeVAE
        The trained TreeVAE model
    device: torch.device
        The device in which to validate the model
    configs: dict
        The config setting for training and validating TreeVAE defined in configs or in the command line
    """
    ESTIMATION_SAMPLES = 1000
    gen_test = get_gen(testset, configs, validation=True, shuffle=False)
    print('\nComputing the log likelihood.... it might take a while.')
    if configs['training']['activation'] == 'sigmoid':
        elbo = np.zeros((len(testset), ESTIMATION_SAMPLES))
        for j in tqdm(range(ESTIMATION_SAMPLES)):
            elbo[:, j] = predict(gen_test, model, device, 'elbo')
            _ = gc.collect()
        elbo_new = elbo[:, :ESTIMATION_SAMPLES]
        log_likel = np.log(1 / ESTIMATION_SAMPLES) + scipy.special.logsumexp(-elbo_new, axis=1)
        marginal_log_likelihood = np.sum(log_likel) / len(testset)
        wandb.log({"test log-likelihood": marginal_log_likelihood})
        print("Test log-likelihood", marginal_log_likelihood)
        output_elbo, output_rec_loss = predict(gen_test, model, device, 'elbo', 'rec_loss')
        print('Test ELBO:', -torch.mean(output_elbo))
        print('Test Reconstruction Loss:', torch.mean(output_rec_loss))

    elif configs['training']['activation'] == 'mse':
        # Correct calculation of ELBO and Loglikelihood for 3channel images without assuming diagonal gaussian for
        # reconstruction
        old_loss = model.loss
        model.loss = loss_reconstruction_cov_mse_eval
        # Note that for comparability to other papers, one might want to add Uniform(0,1) noise to the input images
        # (in 0,255), to go from the discrete to the assumed continuous inputs
        #    x_test_elbo = x_test * 255
        #    x_test_elbo = (x_test_elbo + tfd.Uniform().sample(x_test_elbo.shape)) / 256
        output_elbo, output_rec_loss = predict(gen_test, model, device, 'elbo', 'rec_loss')
        nelbo = torch.mean(output_elbo)
        nelbo_bpd = nelbo / (torch.log(torch.tensor(2)) * configs['training']['inp_shape']) + 8  # Add 8 to account normalizing of inputs
        model.loss = old_loss
        elbo = np.zeros((len(testset), ESTIMATION_SAMPLES))
        for j in range(ESTIMATION_SAMPLES):
            # x_test_elbo = x_test * 255
            # x_test_elbo = (x_test_elbo + tfd.Uniform().sample(x_test_elbo.shape)) / 256
            output_elbo = predict(gen_test, model, device, 'elbo')
            elbo[:, j] = output_elbo
        # Change to bpd
        elbo_new = elbo[:, :ESTIMATION_SAMPLES]
        log_likel = np.log(1 / ESTIMATION_SAMPLES) + scipy.special.logsumexp(-elbo_new, axis=1)
        marginal_log_likelihood = np.sum(log_likel) / len(testset)
        marginal_log_likelihood = marginal_log_likelihood / (
                torch.log(torch.tensor(2)) * configs['training']['inp_shape']) - 8
        wandb.log({"test log-likelihood": marginal_log_likelihood})
        print('Test Log-Likelihood Bound:', marginal_log_likelihood)
        print('Test ELBO:', -nelbo_bpd)
        print('Test Reconstruction Loss:',
              torch.mean(output_rec_loss) / (torch.log(torch.tensor(2)) * configs['training']['inp_shape']) + 8)
        model.loss = old_loss
    else:
        raise NotImplementedError
    return

def get_all_per_sample_losses_simple(model, test_loader, device):
    """
    Simple and reliable function to get per-sample losses for entire dataset
    """
    model.eval()
    all_rec_losses = []
    all_inputs = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Processing batches")):
            inputs = inputs.to(device)
            
            # Forward pass - get the raw outputs
            outputs = model(inputs)
            
            # DEBUG: Check what's available
            if batch_idx == 0:
                print(f"DEBUG: Output keys: {list(outputs.keys())}")
                if 'elbo_samples' in outputs:
                    print(f"DEBUG: elbo_samples shape: {outputs['elbo_samples'].shape}")
            
            # Try to get per-sample losses
            if 'elbo_samples' in outputs:
                # This is the best case - we have per-sample ELBO
                kl_per_sample = outputs['kl_root'] + outputs['kl_decisions'] + outputs['kl_nodes']
                rec_loss_per_sample = outputs['elbo_samples'] - kl_per_sample
                all_rec_losses.extend(rec_loss_per_sample.cpu().numpy())
            else:
                # Fallback: Use batch average for all samples in this batch
                batch_size = inputs.shape[0]
                batch_avg_rec_loss = outputs['rec_loss'].item()
                all_rec_losses.extend([batch_avg_rec_loss] * batch_size)
                print(f"Using batch average for batch {batch_idx}")
            
            # Store inputs for debugging
            all_inputs.extend(inputs.cpu().numpy())
    
    print(f"Processed {len(all_rec_losses)} samples across {batch_idx + 1} batches")
    return np.array(all_rec_losses)

