#!/usr/bin/env python3
"""
Test script to verify the config file is fixed
"""
import yaml

# Load the config to check if seed is now an integer
with open('configs/mnist.yml', 'r') as f:
    configs = yaml.safe_load(f)

print("Config loaded successfully!")
print(f"Seed value: {configs['globals']['seed']}")
print(f"Seed type: {type(configs['globals']['seed'])}")

if isinstance(configs['globals']['seed'], int):
    print("✅ SUCCESS: Seed is now an integer!")
else:
    print("❌ ERROR: Seed is still not an integer")
