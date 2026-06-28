import subprocess
import os
from tqdm import tqdm

program_path = "E:/data/planeExtraction.exe"

input_folder = r"E:data\rn3d\train"
output_folder = r"E:data\rn3d\train_origin"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

threshold = 0.0025 
for filename in tqdm(os.listdir(input_folder), desc='Processing files'):
    if filename.endswith(".txt"):
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename[:-4] + ".txt")
        command = [
            program_path,
            input_file,
            output_file,
            str(threshold)
        ]
        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"Command output: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the command for file {filename}: {e.stderr.strip()}")
            continue

