#!/usr/bin/env python3
"""
Script to build executable from the GUI application using PyInstaller
"""
import os
import sys
import subprocess
from pathlib import Path

def build_executable():
    """Build the executable using PyInstaller"""
    
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # PyInstaller command with options
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",  # Create a single executable file
        "--windowed",  # Hide console window (for GUI apps)
        "--name=DIP_Lab_Image_Studio",  # Name of the executable
        "--icon=app.ico" if os.path.exists("app.ico") else "",  # Icon file if exists
        "--add-data=images:images",  # Include images folder
        "--hidden-import=cv2",  # Ensure OpenCV is included
        "--hidden-import=numpy",  # Ensure NumPy is included
        "--hidden-import=PyQt5",  # Ensure PyQt5 is included
        "--hidden-import=matplotlib",  # Ensure Matplotlib is included
        "--clean",  # Clean PyInstaller cache
        "app_gui.py"  # Main application file
    ]
    
    # Remove empty strings from command
    cmd = [arg for arg in cmd if arg]
    
    print("Building executable with PyInstaller...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run PyInstaller
        result = subprocess.run(cmd, cwd=current_dir, check=True, capture_output=True, text=True)
        print("Build successful!")
        print("\nOutput:")
        print(result.stdout)
        
        # Show location of executable
        if sys.platform.startswith('darwin'):  # macOS
            executable_path = current_dir / "dist" / "DIP_Lab_Image_Studio"
        elif sys.platform.startswith('win'):  # Windows
            executable_path = current_dir / "dist" / "DIP_Lab_Image_Studio.exe"
        else:  # Linux
            executable_path = current_dir / "dist" / "DIP_Lab_Image_Studio"
            
        print(f"\nExecutable created at: {executable_path}")
        print(f"File size: {executable_path.stat().st_size / (1024*1024):.1f} MB")
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        print("Error output:")
        print(e.stderr)
        return False
    
    return True

if __name__ == "__main__":
    build_executable()