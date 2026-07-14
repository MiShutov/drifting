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

        # --- (e) Входные статистики ---
        # Среднее квадратов входного сигнала по каналам
        input_stats = (x ** 2).mean(dim=[2, 3])  # [B, 3]
        features_by_dim = {}
        features_by_dim.setdefault(3, []).append(input_stats)  # размерность 3 (каналы RGB)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)   # [B, C,  H/8,  W/8]
        f2 = self.layer2(f1)  # [B, 2*C, H/16, W/16]
        f3 = self.layer3(f2)  # [B, 4*C, H/32, W/32]
        f4 = self.layer4(f3)  # [B, 8*C, H/64, W/64]

        # Обрабатываем каждую стадию
        for f in [f1, f2, f3, f4]:
            B, C, Hf, Wf = f.shape

            # --- (a) Пространственные векторы ---
            # Каждый пиксель карты признаков как отдельный вектор
            spatial = f.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)  # [B, N, C]
            for i in range(Hf * Wf):
                features_by_dim.setdefault(C, []).append(spatial[:, i, :])  # [B, C]

            # --- (b) Глобальные статистики ---
            mean_global = f.mean(dim=[2, 3])  # [B, C]
            std_global = f.std(dim=[2, 3])    # [B, C]
            features_by_dim.setdefault(C, []).append(mean_global)
            features_by_dim.setdefault(C, []).append(std_global)

            # --- (c) Патчи 2×2 ---
            if Hf >= 2 and Wf >= 2:
                mean_p2, std_p2 = self._get_patch_stats(f, 2)
                # mean_p2: [B, num_patches, C]
                for i in range(mean_p2.shape[1]):
                    features_by_dim.setdefault(C, []).append(mean_p2[:, i, :])
                for i in range(std_p2.shape[1]):
                    features_by_dim.setdefault(C, []).append(std_p2[:, i, :])

            # --- (d) Патчи 4×4 ---
            if Hf >= 4 and Wf >= 4:
                mean_p4, std_p4 = self._get_patch_stats(f, 4)
                # mean_p4: [B, num_patches, C]
                for i in range(mean_p4.shape[1]):
                    features_by_dim.setdefault(C, []).append(mean_p4[:, i, :])
                for i in range(std_p4.shape[1]):
                    features_by_dim.setdefault(C, []).append(std_p4[:, i, :])

        # Группируем по размерности
        grouped_features = []
        for dim, feature_list in features_by_dim.items():
            stacked = torch.stack(feature_list, dim=0)  # [N_features, B, D]
            grouped_features.append(stacked)

        return grouped_features

    def _get_patch_stats(self, f, patch_size):
        """Возвращает среднее и std для каждого патча размера patch_size."""
        B, C, H, W = f.shape
        
        # Используем unfold для извлечения патчей
        patches = f.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # patches: [B, C, num_h, num_w, patch_size, patch_size]
        num_h = H // patch_size
        num_w = W // patch_size
        
        mean_patch = patches.mean(dim=[-2, -1])  # [B, C, num_h, num_w]
        var_patch = patches.var(dim=[-2, -1])    # [B, C, num_h, num_w]
        std_patch = torch.sqrt(var_patch + 1e-8)  # [B, C, num_h, num_w]
        
        # Превращаем в [B, num_patches, C]
        mean_patch = mean_patch.permute(0, 2, 3, 1).reshape(B, num_h * num_w, C)
        std_patch = std_patch.permute(0, 2, 3, 1).reshape(B, num_h * num_w, C)
        
        return mean_patch, std_patch


# import torch
# import torch.nn as nn
# from torchvision import models
# import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint


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
#         if x.min() < 0:
#             x = (x + 1) / 2
#         mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
#         x = (x - mean.to(x.dtype)) / std.to(x.dtype)

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         f1 = self.layer1(x)
#         f2 = self.layer2(f1)
#         f3 = self.layer3(f2)
#         f4 = self.layer4(f3)

#         features_by_dim = {}

#         for f in [f1, f2, f3, f4]:
#             B, C, Hf, Wf = f.shape

#             mean_global = f.mean(dim=[2, 3])  # [B, C]
#             std_global = f.std(dim=[2, 3])    # [B, C]
            
#             features_by_dim.setdefault(C, []).append(mean_global)
#             features_by_dim.setdefault(C, []).append(std_global)

#             # 2. Патчи 2×2 (среднее и std)
#             if Hf >= 2 and Wf >= 2:
#                 mean_p2, std_p2 = self._get_patch_stats(f, 2)
#                 # mean_p2: [B, num_patches, C]
#                 # std_p2:  [B, num_patches, C]
                
#                 # Превращаем в список векторов [B, C]
#                 for i in range(mean_p2.shape[1]):
#                     features_by_dim.setdefault(C, []).append(mean_p2[:, i, :])
#                 for i in range(std_p2.shape[1]):
#                     features_by_dim.setdefault(C, []).append(std_p2[:, i, :])

#         # --- ГРУППИРУЕМ ПО РАЗМЕРНОСТИ ---
#         grouped_features = {}
#         for dim, feature_list in features_by_dim.items():
#             # Конкатенируем все фичи с одинаковой размерностью
#             # feature_list: список [B, D] -> конкатенируем по нулевой оси -> [N_features * B, D]? 
#             # Но нам нужно [N_features, B, D]
            
#             # Сначала стекнем все фичи
#             stacked = torch.stack(feature_list, dim=0)  # [N_features, B, D]
#             grouped_features[dim] = stacked

#         # --- ВОЗВРАЩАЕМ СПИСОК ГРУПП ---
#         # Каждая группа: [N_features, B, D]
#         return list(grouped_features.values())

#     # def forward(self, x):
#     #     # Нормализация входных данных (если x в [-1,1], привести к [0,1])
#     #     if x.min() < 0:
#     #         x = (x + 1) / 2
#     #     mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
#     #     std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
#     #     x = (x - mean.to(x.dtype)) / std.to(x.dtype)

#     #     x = self.conv1(x)
#     #     x = self.bn1(x)
#     #     x = self.relu(x)
#     #     x = self.maxpool(x)  # [B, 64, H/4, W/4]

#     #     f1 = self.layer1(x)   # [B, 64,  H/8,  W/8]
#     #     f2 = self.layer2(f1)  # [B, 128, H/16, W/16]
#     #     f3 = self.layer3(f2)  # [B, 256, H/32, W/32]
#     #     f4 = self.layer4(f3)  # [B, 512, H/64, W/64]

#     #     features = []
#     #     for f in [f1, f2, f3, f4]:
#     #         B, C, Hf, Wf = f.shape

#     #         # Глобальное среднее и std
#     #         features.append(f.mean(dim=[2,3]))  # [B, C]
#     #         features.append(f.std(dim=[2,3]))   # [B, C]

#     #         # Патчи 2×2 (среднее и std)
#     #         if Hf >= 2 and Wf >= 2:
#     #             mean_p2, std_p2 = self._get_patch_stats(f, 2)
#     #             features.extend([mean_p2.reshape(B, -1), std_p2.reshape(B, -1)])
#     #             # Здесь важно: mean_p2 имеет размер [B, num_patches, C]
#     #             # reshape в [B, num_patches * C] даёт один вектор на батч с объединёнными патчами
#     #             # Но лучше оставить как [B, num_patches, C] и добавить каждый патч отдельно
#     #             # как сделано в оригинале:
#     #             for i in range(mean_p2.shape[1]):
#     #                 features.append(mean_p2[:, i, :])
#     #             for i in range(std_p2.shape[1]):
#     #                 features.append(std_p2[:, i, :])

#     #     return features

#         # features = []
#         # # Для каждой стадии извлекаем статистики
#         # for f in [f1, f2, f3, f4]:
#         #     B, C, Hf, Wf = f.shape

#         #     # 1. Все пространственные векторы (каждый пиксель)
#         #     spatial = f.permute(0,2,3,1).reshape(B, Hf*Wf, C)  # [B, N, C]
#         #     for i in range(Hf * Wf):
#         #         features.append(spatial[:, i, :])  # [B, C]

#         #     # 2. Глобальное среднее и стандартное отклонение
#         #     mean_global = f.mean(dim=[2,3])  # [B, C]
#         #     std_global  = f.std(dim=[2,3])   # [B, C]
#         #     features.append(mean_global)
#         #     features.append(std_global)

#         #     # 3. Статистики по патчам 2×2
#         #     if Hf >= 2 and Wf >= 2:
#         #         mean_p2, std_p2 = self._get_patch_stats(f, 2)
#         #         for i in range(mean_p2.shape[1]):
#         #             features.append(mean_p2[:, i, :])
#         #         for i in range(std_p2.shape[1]):
#         #             features.append(std_p2[:, i, :])

#         #     # 4. Статистики по патчам 4×4
#         #     if Hf >= 4 and Wf >= 4:
#         #         mean_p4, std_p4 = self._get_patch_stats(f, 4)
#         #         for i in range(mean_p4.shape[1]):
#         #             features.append(mean_p4[:, i, :])
#         #         for i in range(std_p4.shape[1]):
#         #             features.append(std_p4[:, i, :])

#         # return features  # список тензоров [B, C_i]

#     def _get_patch_stats(self, f, patch_size):
#         """Возвращает среднее и std для каждого патча размера patch_size."""
#         B, C, H, W = f.shape
#         # Извлекаем патчи с помощью unfold
#         patches = f.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
#         # patches: [B, C, num_h, num_w, patch_size, patch_size]
#         num_h = H // patch_size
#         num_w = W // patch_size
#         mean_patch = patches.mean(dim=[-2, -1])       # [B, C, num_h, num_w]
#         var_patch  = patches.var(dim=[-2, -1])        # [B, C, num_h, num_w]
#         std_patch  = torch.sqrt(var_patch + 1e-8)
#         # Преобразуем в [B, num_patches, C]
#         mean_patch = mean_patch.permute(0,2,3,1).reshape(B, num_h*num_w, C)
#         std_patch  = std_patch.permute(0,2,3,1).reshape(B, num_h*num_w, C)
#         return mean_patch, std_patch


# # import torch
# # import torch.nn as nn
# # from torchvision import models
# # import torch.nn.functional as F


# # class MultiScaleFeatureEncoder(nn.Module):
# #     def __init__(self, model_name='resnet34', output_dim=256, freeze=True):
# #         super().__init__()
        
# #         if model_name == 'resnet18':
# #             resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# #         elif model_name == 'resnet34':
# #             resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
# #         elif model_name == 'resnet50':
# #             resnet = models.resnet50(weights=None)
# #             resnet.load_state_dict(
# #                 torch.load("/home/msst/repo/drifting/models/resnet/densecl_r50_coco_1600ep.pth")["state_dict"],
# #                 strict=False
# #             )
        
# #         self.conv1 = resnet.conv1
# #         self.bn1 = resnet.bn1
# #         self.relu = resnet.relu
# #         self.maxpool = resnet.maxpool
# #         self.layer1 = resnet.layer1 
# #         self.layer2 = resnet.layer2  
# #         self.layer3 = resnet.layer3  
# #         self.layer4 = resnet.layer4  
        
# #         if freeze:
# #             for paфram in self.parameters():
# #                 param.requires_grad = False
        
# #         self.proj1 = nn.Linear(256, output_dim)
# #         self.proj2 = nn.Linear(512, output_dim)
# #         self.proj3 = nn.Linear(1024, output_dim)
# #         self.proj4 = nn.Linear(2048, output_dim)
        
# #         # self.norm = nn.LayerNorm(output_dim)
        
# #         self.in_proj = nn.Linear(32, 3)
# #         self.eval()
    
# #     def forward(self, x):
# #         B, C, H, W = x.shape

# #         if x.min() < 0:
# #             x = (x + 1) / 2
        
# #         x = self.conv1(x)
# #         x = self.bn1(x)
# #         x = self.relu(x) # [B, 64, H // 2, W // 2]
# #         # x = self.maxpool(x)  # [B, 64, H // 8, W // 8]

# #         f1 = self.layer1(x)   # [B, 64,  H // 8,  W // 8]
# #         f2 = self.layer2(f1)  # [B, 128, H // 16,  W // 16]
# #         f3 = self.layer3(f2)  # [B, 256, H // 32, W // 32]
# #         f4 = self.layer4(f3)  # [B, 512, H // 64, W // 64]
    
# #         def process_feature(f, proj):
# #             f = F.adaptive_avg_pool2d(f, (1, 1))  # [B, C, 1, 1]
# #             f = f.flatten(1)  # [B, C]
# #             return proj(f)  # [B, output_dim]
        
# #         features = [
# #             process_feature(f1, self.proj1).view(B, -1),
# #             process_feature(f2, self.proj2).view(B, -1),
# #             process_feature(f3, self.proj3).view(B, -1),
# #             process_feature(f4, self.proj4).view(B, -1)
# #             # f2.view(B, -1),
# #             # f3.view(B, -1),
# #             # f4.view(B, -1),
# #         ]
        
# #         return features #torch.cat(features, dim=1)
