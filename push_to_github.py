#!/usr/bin/env python3
"""
Push Code to GitHub

This script pushes all code files to GitHub while excluding large data files
(images, labels, models, cache files, etc.)
"""

import os
import subprocess
import sys
from pathlib import Path


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
    """Create or update .gitignore file."""
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


def get_files_to_add():
    """Get list of files to add to git."""
    # Get all files
    all_files = []
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and common exclusions
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
        
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    
    # Filter files based on extensions we want to include
    code_extensions = {
        '.py', '.sh', '.bat', '.md', '.txt', '.yaml', '.yml', '.json', 
        '.cfg', '.ini', '.conf', '.gitignore', '.dockerfile', '.yml'
    }
    
    # Files to always include
    always_include = {
        'README.md', 'requirements.txt', 'requirements_yolo.txt', 
        'dataset.yaml', 'new_dataset.yaml', '.gitignore'
    }
    
    files_to_add = []
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        
        # Always include specific files
        if file_name in always_include:
            files_to_add.append(file_path)
            continue
        
        # Include code files
        if any(file_path.endswith(ext) for ext in code_extensions):
            # But exclude data directories
            if not any(excluded in file_path for excluded in ['input/', 'output/', 'new_dataset/', 'yolo_dataset/', 'test2/', 'test3/', 'runs/', 'detection_output/']):
                files_to_add.append(file_path)
    
    return files_to_add


def main():
    """Main function."""
    print("=" * 60)
    print("Push Code to GitHub")
    print("=" * 60)
    
    # Check if we're in a git repository
    if not os.path.exists('.git'):
        print("Initializing git repository...")
        if not run_command("git init", "Initializing git"):
            return
    
    # Create/update .gitignore
    create_gitignore()
    
    # Get files to add
    files_to_add = get_files_to_add()
    
    print(f"\nFiles to be added to git ({len(files_to_add)} files):")
    for file_path in sorted(files_to_add):
        print(f"  {file_path}")
    
    # Add files
    if files_to_add:
        files_str = ' '.join(f'"{f}"' for f in files_to_add)
        if not run_command(f"git add {files_str}", "Adding files to git"):
            return
    else:
        print("No files to add")
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
    print("=" * 60)


if __name__ == "__main__":
    main()
