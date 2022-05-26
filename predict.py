import sys

import cv2 as cv
import torch
from torchvision.transforms import ToTensor

from src import config


def main(image_path):
    model = torch.load(config.MODEL_PATH).to(config.DEVICE)

    image = cv.imread(image_path)

    transform = ToTensor()
    image = transform(image)
    image = torch.unsqueeze(image, 0)

    image = image.to(config.DEVICE)

    with torch.no_grad():

        model.eval()
        prediction = model(image)
        print(torch.argmax(prediction))


if __name__ == "__main__":
    image_path = sys.argv[1]
    main(image_path)
