# coding: utf-8
import torchvision.transforms as transforms

def tensor_2_image(img_tensor):
	return transforms.Compose([transforms.ToPILImage()])(img_tensor.cpu())

def image_2_tensor(img_PIL):
	return transforms.Compose([transforms.ToTensor()])(img_PIL)