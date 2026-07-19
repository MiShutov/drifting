import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class PokemonDataset(Dataset):
    def __init__(self, root_dir, dataset_type, img_size=128):
        self.root_dir = Path(root_dir)
            
        if dataset_type == "FACES":
            self.transform = transforms.Compose([
                transforms.Resize((img_size)),
                transforms.CenterCrop((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                )
            ])
        elif dataset_type == "POKEMON":
            self.transform = transforms.Compose([
                # 1. Геометрия (главное)
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.75, 1.0), ratio=(0.9, 1.1)),
                
                # 2. Легкие искажения (чтобы модель не переобучалась на ровные линии)
                transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.05), scale=(0.85, 0.95), fill=255),
                
                # 3. Цвет (мягко)
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
                
                # 4. Шумы / размытие (с малой вероятностью)
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.2),
                transforms.RandomGrayscale(p=0.05),
                
                # 5. Преобразование в тензор
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            raise

        self.image_paths = []
        for pokemon_dir in self.root_dir.iterdir():
            if pokemon_dir.is_dir():
                for img_path in pokemon_dir.glob("*.jpg"):
                    self.image_paths.append(img_path)
        
        print(f"Number of images: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            
        
        return image.to(torch.bfloat16)
