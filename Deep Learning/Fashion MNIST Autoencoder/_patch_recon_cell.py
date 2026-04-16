import nbformat

path = 'Fashion MNIST Autoencoder.ipynb'
nb = nbformat.read(path, as_version=4)
cell = nb.cells[27]
cell['source'] = """# Get batch of test images
test_batch, test_labels = next(iter(test_loader))
test_batch = test_batch.to(DEVICE)

with torch.no_grad():
    recon_batch, _ = model(test_batch)

# Visualize 16 examples (original | reconstruction per sample)
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i in range(16):
    row = i // 4
    col = (i % 4) * 2

    axes[row, col].imshow(test_batch[i].cpu().squeeze(), cmap='gray')
    axes[row, col].set_title(class_names[int(test_labels[i])], fontsize=9)
    axes[row, col].axis('off')

    axes[row, col + 1].imshow(recon_batch[i].cpu().squeeze(), cmap='gray')
    axes[row, col + 1].set_title(f"Recon {class_names[int(test_labels[i])]}", fontsize=9)
    axes[row, col + 1].axis('off')

plt.suptitle('Original (left) vs Reconstructed (right)', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'reconstructions.png', dpi=100, bbox_inches='tight')
plt.show()
print('Reconstruction examples saved.')
"""
nbformat.write(nb, path)
print('CELL_27_PATCHED')
