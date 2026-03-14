import torch
from data_loader import Rescale, ToTensor
import torchvision.transforms as transforms
from PIL import Image
import cv2
import angle
from model import fusionVGG19
from skimage import io
import numpy as np


class ImagemService:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Crie o modelo e o config (ajuste conforme seu config real)
        # Exemplo de config mínimo:
        class Config:
            landmarkNum = 19
            batchSize = 1
            R2 = 41
            image_scale = (800, 640)

        config = Config()
        self.config = config
        image_h, image_w = config.image_scale
        self.transform = transforms.Compose([
            Rescale((image_h, image_w)),
            ToTensor()
        ])

        # Carregue o modelo base (ex: torchvision.models.vgg19)
        import torchvision.models as models

        vgg_model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)

        self.model = fusionVGG19(vgg_model, config).to(self.device)

        # Carregue o state_dict salvo
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, image_path):
        image = io.imread(image_path)
        sample = {'image': image, 'landmarks': np.zeros((self.config.landmarkNum, 2))}  # dummy landmarks
        sample = self.transform(sample)
        input_tensor = sample['image'].unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_tensor)

        coords, _, _ = self.model.getCoordinate(outputs)
        coords_list = coords.squeeze(0).cpu().numpy().tolist()
        points = [angle.Point(x, y) for x, y in coords_list]
        angles = angle.classification(points)
        return coords_list, angles


def desenhar_pontos(img_path, coords, output_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Não foi possível carregar a imagem!")

    h, w = img.shape[:2]

    for x_norm, y_norm in coords:
        x = int(y_norm * w)
        y = int(x_norm * h)
        cv2.circle(img, (x, y), radius=15, color=(0, 0, 255), thickness=-1)

    cv2.imwrite(output_path, img)
    return output_path
