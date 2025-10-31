# Fix for Reconstruction Loss Discrepancy in detailed_probability_analysis.py

## Problem

In the original code, for each sample:
- `sample_summary.csv` showed `final_rec_loss` (e.g., 84.78717)
- `leaf_reconstructions.csv` showed individual leaf `rec_loss` values and `cumulative_prob`
- When computing: **sum(cumulative_prob[i] × rec_loss[i])**, it did NOT equal `final_rec_loss`

Example for sample 0:
- Weighted sum: **0.016** (from leaf_reconstructions)
- Final rec loss: **84.78717** (from sample_summary)
- Ratio: **5183x difference**

## Root Cause

The issue was in `detailed_probability_analysis.py` line 286:

### Original (WRONG):
```python
leaf_rec_loss = torch.nn.functional.mse_loss(x, reconstructions[i]).item()
```

This was using:
1. **MSE loss** instead of the model's actual loss function (BCE for sigmoid)
2. **Default reduction='mean'** which gives average loss per pixel
3. Not summing over all pixels

### Actual Model Loss (loss_reconstruction_binary):
```python
def loss_reconstruction_binary(x, x_decoded_mean, weights):
    x = torch.flatten(x, start_dim=1) 
    x_decoded_mean = [torch.flatten(decoded_leaf, start_dim=1) for decoded_leaf in x_decoded_mean]
    loss = torch.sum(
        torch.stack([weights[i] *
                        F.binary_cross_entropy(input = x_decoded_mean[i], target = x, reduction='none').sum(dim=-1)
                        for i in range(len(x_decoded_mean))], dim=-1), dim=-1)
    return loss
```

This uses:
1. **Binary cross entropy** (for sigmoid activation)
2. **reduction='none'** which gives loss per pixel
3. **.sum(dim=-1)** which sums over all pixels
4. Then multiplies by weight (probability) and sums over all leaves

## Solution

### Fixed Code:
```python
# Flatten input and reconstruction for loss computation
x_flat = torch.flatten(x, start_dim=1)
recon_flat = torch.flatten(reconstructions[i], start_dim=1)

# Use the same loss function as the model
if model.activation == "sigmoid":
    # Binary cross entropy loss (matches loss_reconstruction_binary)
    leaf_rec_loss = torch.nn.functional.binary_cross_entropy(
        input=recon_flat, target=x_flat, reduction='none'
    ).sum(dim=-1).item()
elif model.activation == "mse":
    # MSE loss (matches loss_reconstruction_mse)
    leaf_rec_loss = torch.nn.functional.mse_loss(
        input=recon_flat, target=x_flat, reduction='none'
    ).sum(dim=-1).item()
```

### Key Changes:
1. Uses the **same loss function** as the model (BCE for sigmoid, MSE for mse activation)
2. Uses **reduction='none'** to get loss per pixel
3. Uses **.sum(dim=-1)** to sum over all pixels
4. Flattens both input and reconstruction to match model's processing

### Verification:

The code now also verifies the calculation:
```python
total_weighted_loss += leaves_prob[i].item() * leaf_rec_loss

# Verify that sum(prob[i] * leaf_rec_loss[i]) equals final_rec_loss
diff = abs(total_weighted_loss - final_rec_loss)
if diff > 0.1:
    print(f"Warning: Weighted sum does not match final_rec_loss")
```

## Result

After the fix, for each sample:
```python
sum(cumulative_prob[i] × leaf_rec_loss[i]) == final_rec_loss
```

This ensures that:
- Individual leaf losses in `leaf_reconstructions.csv` correctly represent the contribution of each leaf
- The weighted sum matches the total `final_rec_loss` from `sample_summary.csv`
- The calculation is mathematically consistent with how the model computes reconstruction loss

