import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

class ImageWay:
    def __init__(self, cuda=False):
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = 2048
        self.model, self.extraction_layer = self.get_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.scalar = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        
    def get_model(self):
        model = models.resnet152(pretrained=True)
        layer = model._modules.get('avgpool')

        return model, layer

    def image2vec(self, img, tensor=False):
        image = self.normalize(self.to_tensor(self.scalar(img))).unsqueeze(0).to(self.device)

        print(image.shape)
        embedding = torch.zeros(1, self.layer_output_size, 1, 1)
        def copy_data(m, i, o):
            embedding.copy_(o.data)

        new_image = torch.zeros(1, 3, 224, 224)
        if image.shape[1] >= 3:
            new_image = image[:, :3, :, :]
        else:
            new_image[:, :image.shape[1], :, :] = image[:, :image.shape[1], :, :]
        
        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(new_image)
        h.remove()

        if tensor:
            return embedding
        else:
            return embedding.numpy()[0, :, 0, 0]

    def image_path2vec(self, image_path, tensor=False):
        return self.image2vec(Image.open(image_path), tensor)

image_a = ImageWay().image_path2vec("{address}", tensor=True)
image_b = ImageWay().image_path2vec("{address}", tensor=True)

print(F.cosine_similarity(image_a, image_b))
