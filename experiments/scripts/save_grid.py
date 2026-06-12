from PIL import Image
import matplotlib.pyplot as plt

experiment = "experiment3"

path = f"results/{experiment}/overlays/"

image_paths = [
    f"{path}00000.png", f"{path}00007.png", f"{path}00014.png",
    f"{path}00021.png", f"{path}00029.png", f"{path}00036.png",
    f"{path}00043.png", f"{path}00050.png", f"{path}00059.png",
]

fig, axes = plt.subplots(3, 3, figsize=(9, 9))

for ax, path in zip(axes.flat, image_paths):
    img = Image.open(path)
    ax.imshow(img)
    ax.axis("off")

plt.tight_layout()
plt.savefig(f"{experiment}_grid.png", dpi=300)
plt.show()