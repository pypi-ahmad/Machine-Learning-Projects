"""Fix P49 CIFAR-10 and P19 Fashion MNIST at JSON level."""
import json

ROOT = r"d:\Workspace\Github\Machine-Learning-Projects"

# ── P49: Fix cells 2 (download), 3 (extract), 5 (data_dir) ──
path49 = f"{ROOT}/Machine Learning Project 49 - Cifar 10/04_image_classification_with_CNN(Colab).ipynb"
with open(path49, 'r', encoding='utf-8') as f:
    nb49 = json.load(f)

# Cell 2: download_url — add DATA_DIR setup 
cell2 = nb49['cells'][2]
cell2['source'] = [
    "from pathlib import Path\n",
    "DATA_DIR = Path.cwd().parent / 'data' / 'cifar10'\n",
    "DATA_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "# Download the dataset (tar file) if not already present\n",
    "dataset_url = \"http://files.fast.ai/data/cifar10.tgz\"\n",
    "if not (DATA_DIR / 'train').exists():\n",
    "    download_url(dataset_url, str(DATA_DIR))"
]
print("P49: Fixed cell 2 (download)")

# Cell 3: tarfile extract
cell3 = nb49['cells'][3]
cell3['source'] = [
    "# Extract from archive — the tgz contains a 'cifar10/' folder,\n",
    "# so we extract into DATA_DIR's parent so it lands in DATA_DIR.\n",
    "import tarfile\n",
    "tgz_path = DATA_DIR / 'cifar10.tgz'\n",
    "if tgz_path.exists() and not (DATA_DIR / 'train').exists():\n",
    "    with tarfile.open(str(tgz_path), 'r:gz') as tar:\n",
    "        tar.extractall(path=str(DATA_DIR.parent))"
]
print("P49: Fixed cell 3 (extract)")

# Cell 5: data_dir assignment
cell5 = nb49['cells'][5]
cell5['source'] = [
    "data_dir = str(DATA_DIR)\n",
    "print(os.listdir(data_dir))\n",
    "classes = os.listdir(data_dir + \"/train\")\n",
    "print(classes)"
]
print("P49: Fixed cell 5 (data_dir)")

with open(path49, 'w', encoding='utf-8') as f:
    json.dump(nb49, f, indent=1, ensure_ascii=False)
print("P49: Saved!")

# ── P19: Fix cells 1, 4, 9 ──
path19 = f"{ROOT}/Machine Learning Project 19 - Fashion Mnist Data Analysis Using ML/F_Mnist_model.ipynb"
with open(path19, 'r', encoding='utf-8') as f:
    nb19 = json.load(f)

# Cell 1: dataset = FashionMNIST(...)
cell1 = nb19['cells'][1]
cell1['source'] = [
    "## dataset\n",
    "from pathlib import Path\n",
    "DATA_DIR = Path.cwd().parent / 'data' / 'fashion_mnist'\n",
    "\n",
    "dataset = FashionMNIST(str(DATA_DIR), download=True)"
]
print("P19: Fixed cell 1 (dataset)")

# Cell 4: test_ds
cell4 = nb19['cells'][4]
cell4['source'] = [
    "test_ds = FashionMNIST(str(DATA_DIR), train=False)"
]
print("P19: Fixed cell 4 (test_ds)")

# Cell 9: train_ds + val_ds
cell9 = nb19['cells'][9]
cell9['source'] = [
    "train_ds = FashionMNIST(root=str(DATA_DIR), train=True, transform=train_tfms)\n",
    "val_ds = FashionMNIST(root=str(DATA_DIR), train=False, transform=val_tfms)"
]
print("P19: Fixed cell 9 (train/val)")

with open(path19, 'w', encoding='utf-8') as f:
    json.dump(nb19, f, indent=1, ensure_ascii=False)
print("P19: Saved!")
