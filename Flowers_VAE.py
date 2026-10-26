import torch
import torch.nn as nn

class ResEncodingBlock(nn.Module):
    def __init__(self, channels: list, resolutions: list, k_size: int):
        super().__init__()

        if len(channels) != len(resolutions) or not channels or not resolutions:
            raise ValueError("Las listas deben tener la misma longitud y no estar vacías")
        
        if any(r <= 0 or r % 2 != 0 for r in resolutions):
            raise ValueError("Las resoluciones deben ser positivas y múltiplos de 2")
        
        if k_size % 2 != 1:
            raise ValueError("El tamaño del kernel debe de ser impar")
        for prev, curr in zip(resolutions, resolutions[1:]):
            ratio = prev // curr
            if not (curr == prev or (prev % curr == 0 and ratio & (ratio - 1) == 0)):
                raise ValueError(
            "La resolución debe mantenerse o reducirse en potencias de dos"
            )
    
        for elem in channels[1:]:
            if elem % 4 != 0 or elem <4:
                raise ValueError("Los channels tienen que ser divisibles entre 4")
        
        self.res_transform = None
        if(channels[0] == channels[-1] and resolutions[0] == resolutions[-1]): self.res_transform = nn.Identity()
        else:

            self.res_transform = nn.Conv2d(channels[0], channels[-1], 1, resolutions[0] // resolutions[-1])
        
        len_layers = len(channels)
        layers = []

        for i, j in zip(range(len_layers), range(1, len_layers)):
            new_conv = nn.Conv2d(channels[i], channels[j], k_size, resolutions[i] // resolutions[j], k_size // 2)
            layers.append(new_conv)
            layers.append(nn.ReLU())
            layers.append(nn.GroupNorm(channels[j]// 4, channels[j]))
    
        self.seq = nn.Sequential(*layers)
        
    def forward(self, x):
        y = self.seq(x)
        res = self.res_transform(x)
        return y + res
        

class VAE_Encoder(nn.Module):
    def __init__(self, blocks: int, channels: list, resolutions: list, k_sizes :list, latent_dim: int):
        super().__init__()
        if blocks != len(channels) or blocks != len(resolutions) or blocks != len(k_sizes):
            raise ValueError("Da datos para todos los bloques")
            
        modules = []
        for c, r, k in zip(channels, resolutions, k_sizes):
            modules.append(ResEncodingBlock(c, r, k))
        
        self.encoder = nn.Sequential(*modules)
        
        in_size = channels[-1][-1] * (resolutions[-1][-1] **2)
        self.mu_linear = nn.Linear(in_size, latent_dim)
        self.logvar_linear = nn.Linear(in_size, latent_dim)
        
        
    def forward(self, x):
        
        y = self.encoder(x)
        
        flattened_y = nn.Flatten(start_dim= 1)(y)
        
        logvar = self.logvar_linear(flattened_y)
        mu = self.mu_linear(flattened_y)
        
        return y, logvar, mu
    
    

