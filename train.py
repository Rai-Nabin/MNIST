import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from src import config
from src.model import NeuralNetwork


def visualize_dataset(train_dataset):
    figure = plt.figure(figsize=(8, 8))

    cols, rows = 3, 3

    for i in range(1, cols*rows+1):
        sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
        image, label = train_dataset[sample_idx]
        transform = transforms.ToPILImage()
        image = transform(image)
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(image, cmap="gray")
    plt.show()


def train_one_epoch(model, train_dataloader, loss_function, optimizer):
    model.train()
    total_train_loss = 0
    for data in train_dataloader:
        image, label = data
        image, label = image.to(config.DEVICE), label.to(config.DEVICE)
        prediction = model(image)

        loss = loss_function(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss

    return total_train_loss


def main(train_dataset_path):
    train_dataset = datasets.ImageFolder(
        root=train_dataset_path, transform=ToTensor())
    # visualize_dataset(train_dataset)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = NeuralNetwork().to(config.DEVICE)

    loss_function = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.INIT_LR)
    for i in range(config.NUM_EPOCHS):
        total_train_loss = train_one_epoch(
            model, train_dataloader, loss_function, optimizer)
        print(f'NUM_EPOCHS:{i+1} -> {total_train_loss}')
    torch.save(model, config.MODEL_PATH)


if __name__ == "__main__":
    train_dataset_path = config.TRAIN_DATASET_PATH

    main(train_dataset_path)
