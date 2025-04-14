import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim: int, n_classes: int, output_size: int) -> None:
        """
        Args:
            latent_dim: 噪声向量维度
            n_classes: 条件类别数（3：monster, human, prop）
            output_size: 输出图像展平后元素数量（例如 3 * 16 * 16）
        """
        super().__init__()
        # 条件嵌入：将 n_classes (one-hot向量) 转为 10 维嵌入向量
        self.cond_embed = nn.Linear(n_classes, 10)
        # 拼接后输入维度
        input_dim = latent_dim + 10

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_size),
            nn.Tanh()
        )
    
    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # condition: [batch, n_classes] (one-hot float)
        cond_emb = self.cond_embed(condition)  # [batch, 10]
        x = torch.cat((noise, cond_emb), dim=1)  # [batch, latent_dim + 10]
        out = self.model(x)
        return out.view(x.size(0), 3, 16, 16)
