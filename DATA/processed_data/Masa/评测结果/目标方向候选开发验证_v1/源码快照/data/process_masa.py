"""把本地 Masa（Massachusetts Buildings）原始切片处理成 GOMAA-Geo 需要的格式。

对齐 GOMAA-Geo 官方文档（README "Process the Data"）与脚本逻辑：
  1. data_utils/get_patches.py：每张影像 resize 到 1500x1500（BICUBIC），
     切成 patch_size x patch_size（5x5）网格、每格 300x300，存为 patch_0.jpg~patch_24.jpg。
     patch 顺序为行优先：patch_id = row * 5 + col，与仓库 Sequence 类的
     patch_id//5=行、patch_id%5=列 的寻址方式一致。
  2. data_utils/get_sat_embeddings_sat2cap.py：每个 patch 用 Sat2Cap
     （CLIPVisionModelWithProjection，投影维度 512）编码，预处理为
     Resize(224, BICUBIC) + ToTensor + Normalize(Sat2Cap 均值/方差)，
     每张影像得到 (25, 512) 的嵌入矩阵，按 {'img_i': array} 字典存为 npy，
     文件名与 config.py 的 cfg.data.train/val/test_path 对齐。

输出目录结构：
  DATA/processed_data/Masa/
    patches/{train,val,test}/img_{i}/patch_{j}.jpg
    papr_train_sat_embeds_grid_5.npy
    papr_val_sat_embeds_grid_5.npy
    papr_test_sat_embeds_grid_5.npy
    metadata.csv               # img_id 与原始切片文件名的对应关系
"""
from pathlib import Path
import csv
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "DATA" / "raw_data" / "Masa" / "png"
OUT = ROOT / "DATA" / "processed_data" / "Masa"
MODEL_DIR = ROOT / "models" / "Sat2Cap"

PATCH_SIZE = 5              # GOMAA-Geo config.py: cfg.data.patch_size
TILE = 1500                 # get_patches.py: img.resize((1500, 1500), BICUBIC)
CELL = TILE // PATCH_SIZE   # 300
SPLITS = ["train", "val", "test"]

# get_sat_embeddings_sat2cap.py 的 transform_overhead
MEAN = np.array([0.3670, 0.3827, 0.3338], dtype=np.float32)
STD = np.array([0.2209, 0.1975, 0.1988], dtype=np.float32)


def cut_patches(img_path: Path, save_dir: Path) -> None:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((TILE, TILE), Image.BICUBIC)
    save_dir.mkdir(parents=True, exist_ok=True)
    for i in range(PATCH_SIZE**2):
        row, col = divmod(i, PATCH_SIZE)
        patch = img.crop((col * CELL, row * CELL, (col + 1) * CELL, (row + 1) * CELL))
        patch.save(save_dir / f"patch_{i}.jpg")


def preprocess_patch(patch_path: Path) -> torch.Tensor:
    img = Image.open(patch_path)
    img = img.resize((224, 224), Image.BICUBIC)  # Resize(224, BICUBIC)
    x = np.asarray(img, dtype=np.float32) / 255.0  # ToTensor
    x = (x - MEAN) / STD                            # Normalize
    return torch.from_numpy(x.transpose(2, 0, 1))


def embed_split(model, device: torch.device, patch_root: Path, out_file: Path,
                rows: list) -> None:
    folders = sorted(patch_root.iterdir(), key=lambda p: int(p.name.split("_")[1]))
    embeddings = {}
    with torch.no_grad():
        for folder in tqdm(folders, desc=f"embed {patch_root.name}"):
            patches = [preprocess_patch(folder / f"patch_{j}.jpg")
                       for j in range(PATCH_SIZE**2)]
            batch = torch.stack(patches).to(device)
            # CLIPVisionModelWithProjection -> image_embeds: (25, 512)
            preds = model(batch).image_embeds.cpu().numpy().astype(np.float32)
            embeddings[folder.name] = preds
            rows.append({"split": patch_root.name, "img_id": folder.name,
                         "source_tile": source_of[folder.name + "@" + patch_root.name]})
    np.save(out_file, embeddings)
    print(f"saved {out_file.name}: {len(embeddings)} images, "
          f"{next(iter(embeddings.values())).shape} per image")


if __name__ == "__main__":
    # 只处理 patch 阶段
    if "--patches-only" in sys.argv:
        for split in SPLITS:
            tiles = sorted((RAW / split).glob("*.png"))
            for i, tile in enumerate(tiles):
                cut_patches(tile, OUT / "patches" / split / f"img_{i}")
            print(f"{split}: {len(tiles)} tiles -> patches done")
        sys.exit(0)

    # 记录 img_id -> 原始切片名（切 patch 时顺便登记）
    source_of = {}
    for split in SPLITS:
        tiles = sorted((RAW / split).glob("*.png"))
        for i, tile in enumerate(tiles):
            cut_patches(tile, OUT / "patches" / split / f"img_{i}")
            source_of[f"img_{i}@{split}"] = tile.name
        print(f"{split}: {len(tiles)} tiles cut into {PATCH_SIZE**2} patches each")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CLIPVisionModelWithProjection.from_pretrained(str(MODEL_DIR)).to(device).eval()

    rows = []
    for split in SPLITS:
        embed_split(model, device, OUT / "patches" / split,
                    OUT / f"papr_{split}_sat_embeds_grid_5.npy", rows)

    with open(OUT / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "img_id", "source_tile"])
        writer.writeheader()
        writer.writerows(rows)
    print("metadata.csv written")
