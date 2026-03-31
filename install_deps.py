#!/usr/bin/env python3
"""Script to install missing dependencies for the RAG system."""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install missing dependencies."""
    print("🔧 Installing missing dependencies for RAG system...")
    print()
    
    # Core dependencies that were missing
    dependencies = [
        "torch>=2.0",
        "torchvision>=0.15", 
        "transformers>=4.36",
    ]
    
    failed_packages = []
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        if not install_package(dep):
            failed_packages.append(dep)
        print()
    
    if failed_packages:
        print(f"⚠️  Failed to install: {', '.join(failed_packages)}")
        print("You may need to install these manually.")
        return 1
    else:
        print("🎉 All dependencies installed successfully!")
        print("You can now restart your Streamlit app.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
