#!/usr/bin/env python3
"""
Simple test to verify model loading works
"""

import torch
import yaml
from pathlib import Path
from models.model import TreeVAE

def test_model_loading():
    model_path = "/scratch/bep91wab/projectTreeVAE_Pytorch/treevae/models/experiments/mnist/20250922-030449_a4181"
    
    # Load config with custom loader to handle pathlib.PosixPath
    def pathlib_constructor(loader, node):
        value = loader.construct_sequence(node)
        return Path(*value)
    
    class CustomLoader(yaml.Loader):
        pass
    
    CustomLoader.add_constructor('tag:yaml.org,2002:python/object/apply:pathlib.PosixPath', pathlib_constructor)
    
    config_file = Path(model_path) / 'config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=CustomLoader)
    
    # Create model
    model = TreeVAE(**config['training'])
    
    # Load weights
    model_file = Path(model_path) / 'model_weights.pt'
    state_dict = torch.load(model_file, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    print(f"✅ Model loaded successfully!")
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    print(f"Model depth: {model.compute_depth()}")
    print(f"Number of leaves: {len(model.compute_leaves())}")
    
    return True

if __name__ == "__main__":
    test_model_loading()




