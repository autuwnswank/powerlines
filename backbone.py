import torch
import torch.nn as nn
from torchvision.models import resnet18

# 1. Загружаем бэкбон
model = resnet18(weights='IMAGENET1K_V1')
model.eval()

# 2. Словарь для хранения активаций
activations = {}

# 3. Функция-хук: что делать, когда слой сработал
def get_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# 4. Вешаем хуки на интересные слои (например, layer3 и layer4)
model.layer3.register_forward_hook(get_activation('layer3'))
model.layer4.register_forward_hook(get_activation('layer4'))

# 5. Прогоняем картинку
x = torch.randn(1, 3, 224, 224)  # здесь будет ваш тензор изображения
with torch.no_grad():
    _ = model(x)

# 6. Смотрим, что внутри
for name, act in activations.items():
    print(f"{name}: shape={act.shape}, mean={act.mean():.4f}, max={act.max():.4f}")