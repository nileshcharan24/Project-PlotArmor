"""
Download TinyStories dataset from HuggingFace.
Split into train/val and save to research/data/.
"""

from datasets import load_dataset
import os


def main():
    # Load the dataset
    print("Loading TinyStories dataset...")
    dataset = load_dataset('roneneldan/TinyStories')

    # Extract text
    train_texts = dataset['train']['text']
    val_texts = dataset['validation']['text']

    # Combine all text
    all_train_text = '\n'.join(train_texts)
    all_val_text = '\n'.join(val_texts)

    # Ensure data directory exists
    data_dir = 'research/data'
    os.makedirs(data_dir, exist_ok=True)

    # Save to files
    train_file = os.path.join(data_dir, 'tinystories_train.txt')
    val_file = os.path.join(data_dir, 'tinystories_val.txt')

    print("Saving train data...")
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(all_train_text)

    print("Saving val data...")
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write(all_val_text)

    # Print sizes
    train_size = os.path.getsize(train_file) / (1024 * 1024)  # MB
    val_size = os.path.getsize(val_file) / (1024 * 1024)  # MB

    print(f"Train file: {train_file}, Size: {train_size:.2f} MB")
    print(f"Val file: {val_file}, Size: {val_size:.2f} MB")
    print("Download complete!")


if __name__ == "__main__":
    main()