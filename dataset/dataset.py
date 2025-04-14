import os
import csv
from PIL import Image
from torch.utils.data import Dataset
import torch

class PixelDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file: CSV 文件路径，例如 "resources/pixel_dataset/labels.csv"
            root_dir: 图片所在目录，例如 "resources/pixel_dataset/images"
            transform: 图像预处理
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # 每个元素: (img_filename, label)

        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 从 CSV 获取图片相对路径（CSV 中写的是 jpg 后缀）
                img_rel_path = row["Image Path"].strip()
                # 取文件名，例如 "image_15519.jpg"
                img_filename = os.path.basename(img_rel_path)
                
                full_path = os.path.join(root_dir, img_filename)
                # 如果文件不存在，尝试更换扩展名: 如果扩展名为 .jpg，则换成 .JPEG
                if not os.path.exists(full_path):
                    base, ext = os.path.splitext(img_filename)
                    if ext.lower() == ".jpg":
                        alternative_name = base + ".JPEG"
                        full_path_alt = os.path.join(root_dir, alternative_name)
                        if os.path.exists(full_path_alt):
                            print(f"Notice: Using alternative filename: {alternative_name}")
                            img_filename = alternative_name
                            full_path = full_path_alt
                
                if os.path.exists(full_path):
                    label_str = row["Label"].strip().strip("[]")
                    parts = label_str.replace(",", " ").split()
                    try:
                        values = [float(x) for x in parts]
                    except Exception as e:
                        print(f"Warning: Failed to convert label {label_str}: {e}. Skipping this sample.")
                        continue
                    if len(values) < 5:
                        print(f"Warning: Label 不够5个数字, 跳过: {label_str}")
                        continue
                    # 用 parse_label 函数将 5 个数字转换为 3 类标签
                    int_label = self.parse_label(values)
                    self.samples.append((img_filename, int_label))
                else:
                    print(f"Warning: File not found: {full_path}. Skipping this sample.")
        print(f"Total valid samples: {len(self.samples)}")

    def parse_label(self, values):
        """
        将 5 个数字 [v1, v2, v3, v4, v5] 转换为 3 类标签：
          - 如果 v1 == 1，则返回 0（角色）
          - 否则如果 v2 == 1，则返回 1（monster）
          - 否则如果 v3 == 1 或 v4 == 1，则返回 2（物品）
          - 否则返回 2
        """
        v1, v2, v3, v4, _ = values[:5]
        if v1 == 1.0:
            return 0  # 角色
        elif v2 == 1.0:
            return 1  # monster
        elif v3 == 1.0 or v4 == 1.0:
            return 2  # 物品
        else:
            return 2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_filename, int_label = self.samples[idx]
        full_path = os.path.join(self.root_dir, img_filename)
        img = Image.open(full_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # 转换标签为 one-hot 向量（3 维），例如:
        one_hot_label = torch.zeros(3, dtype=torch.float32)
        one_hot_label[int_label] = 1.0
        return img, one_hot_label
