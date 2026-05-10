from pathlib import Path

train_path = Path("dataset/Training")
test_path = Path("dataset/Testing")

print("=" * 40)
print("TRAINING DATASET")
print("=" * 40)

for folder in train_path.iterdir():
    if folder.is_dir():
        count = len(list(folder.glob("*")))
        print(f"{folder.name}: {count} images")

print("\n" + "=" * 40)
print("TESTING DATASET")
print("=" * 40)

for folder in test_path.iterdir():
    if folder.is_dir():
        count = len(list(folder.glob("*")))
        print(f"{folder.name}: {count} images")