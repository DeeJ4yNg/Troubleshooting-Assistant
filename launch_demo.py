#!/usr/bin/env python3
"""
Launcher for the Windows OS Troubleshooting Agent Rich UI Demo
"""

import subprocess
import sys

def main():
    print("🚀 Launching Windows OS Troubleshooting Agent Demo with Rich UI...")
    print("Please wait while the demo runs...\n")
    
    try:
        # Run the demo script
        result = subprocess.run([sys.executable, "demo_rich_ui.py"], 
                              capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ Demo completed successfully!")
            print("\n📋 Output:")
            print(result.stdout)
        else:
            print("❌ Demo encountered an error:")
            print(result.stderr)
            
    except FileNotFoundError:
        print("❌ Error: demo_rich_ui.py not found!")
        print("Please make sure you're running this from the project directory.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()