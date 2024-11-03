import torch
from PIL import Image
from u2net_fast.remover import Remover
# Initialize the Remover with the path to the model weights
remover = Remover(model_path='os.getcwd()}/models/u2net.pth')
remover.batch_remove_background("mask_tool\code\image","mask_tool\output")
