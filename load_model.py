from torch import nn
from torchvision import models
import torchvision.models as models
from collections import OrderedDict

def load_model(path):
	print('Loading model from checkpoint....')
	if not torch.cuda.is_available():
		checkpoint = torch.load(path, map_location='cpu')
	else:
		checkpoint = torch.load(path)

	if 'h1' in checkpoint:
		h1 = checkpoint['h1']

	else:
		h1 = 120

	if checkpoint['transfer_model'] == 'densenet121':
		model = models.densenet121(pretrained=True)
		for param in model.parameters():
			param.requires_grad = False
			classifier = nn.Sequential(OrderedDict([
				('fc1', nn.Linear(1024, h1)), ('relu', nn.ReLU()),
				('relu', nn.ReLU()),
				('dropout', nn.Dropout(p=0.2)),
				('fc2', nn.Linear(h1, 102)),
				('output', nn.LogSoftmax(dim=1))
				]))
			model.classifier = classifier

	else: 
		model = models.resnet18(pretrained=True)
		for param in model.parameters():
			param.requires_grad = False
		classifier = nn.Sequential(OrderedDict([
				('fc1', nn.Linear(512, h1)), ('relu', nn.ReLU()),
				('relu', nn.ReLU()),
				('dropout', nn.Dropout(p=0.2)),
				('fc2', nn.Linear(h1, 102)),
				('output', nn.LogSoftmax(dim=1))
				]))
		model.fc = classifier

	model.load_state_dict(checkpoint['state_dict'])
	model.idx_to_class = checkpoint['idx_to_class']
	return model
