#!/usr/bin/env python3
"""
Simple GitHub Push - Code Only

This script pushes only code files to GitHub, excluding all data files.
"""

import os
import subprocess
import sys


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def create_gitignore():
    """Create comprehensive .gitignore file."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Data files (large)
*.jpg
*.jpeg
*.png
*.bmp
*.tiff
*.gif
*.mp4
*.avi
*.mov
*.mkv
*.zip
*.rar
*.7z
*.tar
*.gz

# Labels and annotations
*.txt
*.xml
*.json
*.csv

# Model files
*.pt
*.pth
*.onnx
*.engine
*.trt
*.h5
*.pb
*.pkl
*.joblib

# Cache files
*.cache
labels.cache
images.cache

# Training outputs
runs/
wandb/
logs/
checkpoints/

# Dataset directories
input/
output/
new_dataset/
yolo_dataset/
test2/
test3/
detection_output/
realtime_detection_output/
pi_dataset/
pi_augmented/
pi_transfer_package/

# Temporary files
temp/
tmp/
*.tmp
*.log

# Jupyter
.ipynb_checkpoints/

# PyTorch
*.pth.tar

# YOLO specific
*.yaml
args.yaml
results.csv
results.png
confusion_matrix*.png
Box*.png
train_batch*.jpg
val_batch*.jpg
labels.jpg

# Keep important config files
!dataset.yaml
!new_dataset.yaml
!requirements*.txt
!README.md
!*.py
!*.sh
!*.bat
!*.md
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("Created/updated .gitignore file")


def main():
    """Main function."""
    print("=" * 60)
    print("Simple GitHub Push - Code Only")
    print("=" * 60)
    
    # Check if we're in a git repository
    if not os.path.exists('.git'):
        print("Initializing git repository...")
        if not run_command("git init", "Initializing git"):
            return
    
    # Create/update .gitignore
    create_gitignore()
    
    # Add all files (gitignore will handle exclusions)
    if not run_command("git add .", "Adding files to git"):
        return
    
    # Check git status
    run_command("git status", "Checking git status")
    
    # Commit
    commit_message = "Add YOLO12 object detection pipeline with data augmentation and real-time detection"
    if not run_command(f'git commit -m "{commit_message}"', "Committing changes"):
        return
    
    # Check if remote exists
    result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True)
    if not result.stdout.strip():
        print("\nNo remote repository found.")
        repo_url = input("Enter your GitHub repository URL (or press Enter to skip): ").strip()
        if repo_url:
            if not run_command(f"git remote add origin {repo_url}", "Adding remote origin"):
                return
        else:
            print("Skipping remote setup. You can add it later with:")
            print("git remote add origin <your-repo-url>")
            return
    
    # Push to GitHub
    print("\nPushing to GitHub...")
    if not run_command("git push -u origin main", "Pushing to main branch"):
        # Try master branch if main fails
        if not run_command("git push -u origin master", "Pushing to master branch"):
            print("Failed to push. You may need to:")
            print("1. Set up your GitHub repository")
            print("2. Configure git credentials")
            print("3. Check your internet connection")
            return
    
    print("\n" + "=" * 60)
    print("SUCCESS! Code pushed to GitHub")
    print("=" * 60)
    print("Excluded files:")
    print("  - Images (*.jpg, *.png, etc.)")
    print("  - Labels (*.txt)")
    print("  - Model files (*.pt, *.pth)")
    print("  - Cache files (*.cache)")
    print("  - Training outputs (runs/)")
    print("  - Dataset directories")
    print("  - Detection output files")
    print("=" * 60)


if __name__ == "__main__":
    main()
