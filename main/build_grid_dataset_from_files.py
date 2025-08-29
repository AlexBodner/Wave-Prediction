import os
import re
from pathlib import Path

def extract_timestamp(filename):
    """Extrae el timestamp del nombre de archivo 'grid_<timestamp>.npy'"""
    match = re.search(r'grid_(\d+)\.npy', filename)
    return int(match.group(1)) if match else None

def build_dataset(root_dir='./numpy_grids'):
    sequences = []

    root_path = Path(root_dir)
    for recording_dir in sorted(root_path.iterdir()):
        if not recording_dir.is_dir():
            continue

        grid_files = [f for f in recording_dir.iterdir() if f.name.startswith('grid_') and f.suffix == '.npy']
        grid_files = sorted(grid_files, key=lambda f: extract_timestamp(f.name))

        # Formar ventanas deslizantes de 4 elementos (3 inputs + 1 target)
        for i in range(len(grid_files) - 3):
            input1 = grid_files[i]
            input2 = grid_files[i+1]
            input3 = grid_files[i+2]
            target = grid_files[i+3]

            sequences.append((str(input1), str(input2), str(input3), str(target)))

    return sequences

if __name__ == "__main__":
    dataset = build_dataset()
    print(f"Total sequences found: {len(dataset)}")
    # Ejemplo de output
    for i, (a, b, c, d) in enumerate(dataset[:5]):
        print(f"[{i}] Inputs:\n  {a}\n  {b}\n  {c}\nTarget:\n  {d}")
