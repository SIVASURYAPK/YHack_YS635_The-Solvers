import os
import time
import subprocess

dataset = "tanlikesmath/diabetic-retinopathy-resized"
command = f"python -m kaggle datasets download -d {dataset} --unzip"

attempt = 1
while True:
    print(f"\n--- Starting download attempt {attempt} ---")
    result = subprocess.run(command, shell=True)
    
    # Check if download succeeded (exit code 0)
    if result.returncode == 0:
        print("\n✅ Download and extraction completed successfully!")
        break
    
    attempt += 1
    print("\n⚠️ Connection dropped. Retrying in 5 seconds to resume progress...")
    time.sleep(5)