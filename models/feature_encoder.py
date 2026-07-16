import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class MultiScaleFeatureEncoder(nn.Module):
    def __init__(self, model_name='resnet34', freeze=False):
        super().__init__()
        
        if model_name == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet50':
            resnet = models.resnet50(weights=None)
            state_dict = torch.load("/home/msst/repo/drifting/models/resnet/densecl_r50_coco_1600ep.pth")["state_dict"]
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            resnet.load_state_dict(new_state_dict, strict=False)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        
        # Нормализация входных данных
        if x.min() < 0:
            x = (x + 1) / 2

        features_by_dim = {}

        # --- (e) Входные статистики ---
        input_stats = (x ** 2).mean(dim=[2, 3])  # [B, 3]
        features_by_dim.setdefault(3, []).append(input_stats.unsqueeze(0))  # [1, B, 3]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)   # [B, C,  H/8,  W/8]
        f2 = self.layer2(f1)  # [B, 2*C, H/16, W/16]
        f3 = self.layer3(f2)  # [B, 4*C, H/32, W/32]
        f4 = self.layer4(f3)  # [B, 8*C, H/64, W/64]

        for f in [f1, f2, f3, f4]:
            B_f, C, Hf, Wf = f.shape
            
            # --- (a) Пространственные векторы ---
            spatial = f.permute(0, 2, 3, 1).reshape(B_f, Hf * Wf, C)  # [B, N, C]
            spatial_features = spatial.permute(1, 0, 2)  # [N, B, C]
            features_by_dim.setdefault(C, []).append(spatial_features)
            
            # --- (b) Глобальные статистики ---
            mean_global = f.mean(dim=[2, 3])  # [B, C]
            std_global = f.std(dim=[2, 3])    # [B, C]
            features_by_dim.setdefault(C, []).append(mean_global.unsqueeze(0))  # [1, B, C]
            features_by_dim.setdefault(C, []).append(std_global.unsqueeze(0))    # [1, B, C]
            
            # --- (c) Патчи 2×2 ---
            if Hf >= 2 and Wf >= 2:
                mean_p2, std_p2 = self._get_patch_stats(f, 2)
                features_by_dim.setdefault(C, []).append(mean_p2.permute(1, 0, 2))  # [num_patches, B, C]
                features_by_dim.setdefault(C, []).append(std_p2.permute(1, 0, 2))
            
            # --- (d) Патчи 4×4 ---
            if Hf >= 4 and Wf >= 4:
                mean_p4, std_p4 = self._get_patch_stats(f, 4)
                features_by_dim.setdefault(C, []).append(mean_p4.permute(1, 0, 2))
                features_by_dim.setdefault(C, []).append(std_p4.permute(1, 0, 2))
        
        # Конкатенация по размерности N_features
        grouped_features = []
        for dim, feature_list in features_by_dim.items():
            stacked = torch.cat(feature_list, dim=0)  # [total_N_features, B, D]
            grouped_features.append(stacked)
        
        return grouped_features

    @torch.compile()
    def _get_patch_stats(self, f, patch_size):
        """Возвращает среднее и std для каждого патча размера patch_size."""
        B, C, H, W = f.shape
        
        patches = f.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # patches: [B, C, num_h, num_w, patch_size, patch_size]
        
        mean_patch = patches.mean(dim=[-2, -1])  # [B, C, num_h, num_w]
        var_patch = patches.var(dim=[-2, -1])    # [B, C, num_h, num_w]
        std_patch = torch.sqrt(var_patch + 1e-8)  # [B, C, num_h, num_w]
        
        # Превращаем в [B, num_patches, C]
        mean_patch = mean_patch.permute(0, 2, 3, 1).reshape(B, -1, C)
        std_patch = std_patch.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        return mean_patch, std_patch



# import torch
# import torch.nn as nn
# from torchvision import models
# import torch.nn.functional as F


# class MultiScaleFeatureEncoder(nn.Module):
#     def __init__(self, model_name='resnet34', freeze=False):
#         super().__init__()
        
#         if model_name == 'resnet18':
#             resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#         elif model_name == 'resnet34':
#             resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
#         elif model_name == 'resnet50':
#             resnet = models.resnet50(weights=None)
#             state_dict = torch.load("/home/msst/repo/drifting/models/resnet/densecl_r50_coco_1600ep.pth")["state_dict"]
#             new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#             resnet.load_state_dict(new_state_dict, strict=False)
#         else:
#             raise ValueError(f"Unknown model_name: {model_name}")
        
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool
#         self.layer1 = resnet.layer1
#         self.layer2 = resnet.layer2
#         self.layer3 = resnet.layer3
#         self.layer4 = resnet.layer4

#         if freeze:
#             for param in self.parameters():
#                 param.requires_grad = False

#     def forward(self, x):
#         B = x.shape[0]
        
#         # Нормализация входных данных
#         if x.min() < 0:
#             x = (x + 1) / 2

#         features_by_dim = {}

#         # # --- (e) Входные статистики ---
#         # # Среднее квадратов входного сигнала по каналам
#         # input_stats = (x ** 2).mean(dim=[2, 3])  # [B, 3]
#         # features_by_dim.setdefault(3, []).append(input_stats)  # размерность 3 (каналы RGB)

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         # x = self.maxpool(x) # Removed

#         f1 = self.layer1(x)   # [B, C,  H/8,  W/8]
#         f2 = self.layer2(f1)  # [B, 2*C, H/16, W/16]
#         f3 = self.layer3(f2)  # [B, 4*C, H/32, W/32]
#         f4 = self.layer4(f3)  # [B, 8*C, H/64, W/64]

#         for f in [f1, f2, f3, f4]:  # Removed
#         # for f in [f3, f4]: 
#             B, C, Hf, Wf = f.shape

#             # --- (a) Пространственные векторы ---
#             # Каждый пиксель карты признаков как отдельный вектор
#             spatial = f.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)  # [B, N, C]
#             for i in range(Hf * Wf):
#                 features_by_dim.setdefault(C, []).append(spatial[:, i, :])  # [B, C]

#             # --- (b) Глобальные статистики ---
#             mean_global = f.mean(dim=[2, 3])  # [B, C]
#             std_global = f.std(dim=[2, 3])    # [B, C]
#             features_by_dim.setdefault(C, []).append(mean_global)
#             features_by_dim.setdefault(C, []).append(std_global)

#             # --- (c) Патчи 2×2 ---
#             if Hf >= 2 and Wf >= 2:
#                 mean_p2, std_p2 = self._get_patch_stats(f, 2)
#                 # mean_p2: [B, num_patches, C]
#                 for i in range(mean_p2.shape[1]):
#                     features_by_dim.setdefault(C, []).append(mean_p2[:, i, :])
#                 for i in range(std_p2.shape[1]):
#                     features_by_dim.setdefault(C, []).append(std_p2[:, i, :])

#             # --- (d) Патчи 4×4 ---
#             if Hf >= 4 and Wf >= 4:
#                 mean_p4, std_p4 = self._get_patch_stats(f, 4)
#                 # mean_p4: [B, num_patches, C]
#                 for i in range(mean_p4.shape[1]):
#                     features_by_dim.setdefault(C, []).append(mean_p4[:, i, :])
#                 for i in range(std_p4.shape[1]):
#                     features_by_dim.setdefault(C, []).append(std_p4[:, i, :])

#         # Группируем по размерности
#         grouped_features = []
#         for dim, feature_list in features_by_dim.items():
#             stacked = torch.stack(feature_list, dim=0)  # [N_features, B, D]
#             grouped_features.append(stacked)

#         return grouped_features

#     @torch.compile()
#     def _get_patch_stats(self, f, patch_size):
#         """Возвращает среднее и std для каждого патча размера patch_size."""
#         B, C, H, W = f.shape
        
#         # Используем unfold для извлечения патчей
#         patches = f.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
#         # patches: [B, C, num_h, num_w, patch_size, patch_size]
#         num_h = H // patch_size
#         num_w = W // patch_size
        
#         mean_patch = patches.mean(dim=[-2, -1])  # [B, C, num_h, num_w]
#         var_patch = patches.var(dim=[-2, -1])    # [B, C, num_h, num_w]
#         std_patch = torch.sqrt(var_patch + 1e-8)  # [B, C, num_h, num_w]
        
#         # Превращаем в [B, num_patches, C]
#         mean_patch = mean_patch.permute(0, 2, 3, 1).reshape(B, num_h * num_w, C)
#         std_patch = std_patch.permute(0, 2, 3, 1).reshape(B, num_h * num_w, C)
        
#         return mean_patch, std_patch
