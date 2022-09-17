import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', './best.pt')  # or yolov5m, yolov5l, yolov5x, custom
model.conf = 0.1
model.iou = 0.3

# Images
img = '/home/tim/Downloads/test3.jpeg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)
results.render()

results.crop(save=True, save_dir='./crop')

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
