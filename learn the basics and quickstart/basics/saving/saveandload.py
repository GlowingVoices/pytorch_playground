import torch
import torchvision.models as models


model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
model.eval()

#The file extension for this project is '.pth'

model = models.vgg16()

model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

#reloading the weights in the shape of the model variable
torch.save(model, 'model.pth')
model = torch.load('model.pth')
