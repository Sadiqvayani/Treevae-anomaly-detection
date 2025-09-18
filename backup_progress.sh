#!/bin/bash

# Backup script for TreeVAE experiments
# Usage: ./backup_progress.sh

echo "Creating backup of current progress..."

# Create backup directory with timestamp
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup code changes
echo "Backing up code files..."
cp -r configs/ "$BACKUP_DIR/"
cp -r models/ "$BACKUP_DIR/"
cp -r utils/ "$BACKUP_DIR/"
cp -r train/ "$BACKUP_DIR/"
cp main.py "$BACKUP_DIR/"

# Backup recent output files
echo "Backing up output files..."
cp treevae_*_out.txt "$BACKUP_DIR/" 2>/dev/null || echo "No output files found"

# Backup results
echo "Backing up results..."
cp -r results/ "$BACKUP_DIR/" 2>/dev/null || echo "No results directory found"

# Create git snapshot
echo "Creating git snapshot..."
cd "$BACKUP_DIR"
git init
git add .
git commit -m "Backup snapshot $(date)"

echo "Backup completed in directory: $BACKUP_DIR"
echo "To restore: cp -r $BACKUP_DIR/* ."
