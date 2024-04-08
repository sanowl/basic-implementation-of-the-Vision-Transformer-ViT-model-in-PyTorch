import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio=4.0, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=qkv_bias)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        q = k = v = self.norm1(x)
        x = x + self.attn(q, k, v)[0]
        x = x + self.mlp(self.norm2(x))
        return x

if __name__ == '__main__':
    # Define the model parameters
    img_size = 224
    patch_size = 16
    in_channels = 3
    num_classes = 10
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_ratio = 4.0
    qkv_bias = False
    dropout = 0.1

    # Create an instance of the Vision Transformer model
    model = VisionTransformer(
        img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads,
        mlp_ratio, qkv_bias, dropout
    )

    # Generate a random input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, in_channels, img_size, img_size)

    # Forward pass through the model
    output = model(input_tensor)

    # Print the output shape
    print("Output shape:", output.shape)