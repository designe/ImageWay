import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

model_info = {
    'resnet152' : models.resnet152(pretrained=True),
    'resnet18' : models.resnet18(pretrained=True),
    'inception' : models.inception_v3(pretrained=True),
    'resnext101_32x8d_wsl' : torch.hub.load('facebookresearch/WSL-images', 'resnext101_32x8d_wsl'),
    'resnext101_32x16d_wsl' : torch.hub.load('facebookresearch/WSL-images', 'resnext101_32x16d_wsl'),
    'resnext101_32x32d_wsl' : torch.hub.load('facebookresearch/WSL-images', 'resnext101_32x32d_wsl'),
    'resnext101_32x48d_wsl' : torch.hub.load('facebookresearch/WSL-images', 'resnext101_32x48d_wsl')
    }

class ImageWay:
    def __init__(self, cuda=False):
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = 1000
        self.model, self.extraction_layer = self.get_model('resnext101_32x48d_wsl')
        self.model = self.model.to(self.device)
        self.model.eval()
        self.scalar = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        
    def get_model(self, model_name='resnet152'):
        model = model_info[model_name]
        layer = model._modules.get('fc')
        return model, layer

    def image2vec(self, img, tensor=False):
        image = img.convert('RGB')
        image = self.normalize(self.to_tensor(self.scalar(image))).unsqueeze(0).to(self.device)

        embedding = torch.zeros(1, self.layer_output_size)
        # embedding = torch.zeros(1, self.layer_output_size, 1, 1)
        def copy_data(m, i, o):
            embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        if tensor:
            return embedding
        else:
            return embedding.numpy()[0, :, 0, 0]

    def image_path2vec(self, image_path, tensor=False):
        return self.image2vec(Image.open(image_path), tensor)

#image_a = ImageWay().image_path2vec("../analogwatch/20_face_76.png", tensor=True)
#image_b = ImageWay().image_path2vec("../analogwatch/20_face_77.png", tensor=True)
#image_c = ImageWay().image_path2vec("../analogwatch/20_face_78.png", tensor=True)

#print(F.cosine_similarity(image_a.view(1, -1), image_b.view(1, -1)))
#print(F.cosine_similarity(image_a.view(1, -1), image_c.view(1, -1)))
#print(F.cosine_similarity(image_b.view(1, -1), image_c.view(1, -1)))

#print(F.pairwise_distance(image_a.view(1, -1), image_b.view(1, -1)))
#print(F.pairwise_distance(image_a.view(1, -1), image_c.view(1, -1)))
#print(F.pairwise_distance(image_b.view(1, -1), image_c.view(1, -1)))

