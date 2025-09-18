#!/bin/bash

# Script to run TreeVAE with different anomaly digits
# Usage: ./run_anomaly_experiments.sh

echo "Starting anomaly detection experiments..."

# Create results directory
mkdir -p results/experiments

# Function to run experiment with specific anomaly digit
run_experiment() {
    local anomaly_digit=$1
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    echo "Running experiment with anomaly digit: $anomaly_digit"
    
    # Update config file
    sed -i "s/anomaly_digit: [0-9]*/anomaly_digit: $anomaly_digit/" configs/mnist.yml
    
    # Run the experiment
    python main.py > "results/experiments/anomaly_digit_${anomaly_digit}_${timestamp}.txt" 2>&1
    
    # Save config for this experiment
    cp configs/mnist.yml "results/experiments/config_anomaly_digit_${anomaly_digit}_${timestamp}.yml"
    
    echo "Experiment with anomaly digit $anomaly_digit completed. Output saved to results/experiments/"
}

# Run experiments for different anomaly digits
for digit in 0 1 2 3 4 5 6 7 8 9; do
    run_experiment $digit
    echo "Waiting 5 seconds before next experiment..."
    sleep 5
done

echo "All experiments completed!"
