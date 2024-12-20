import torch
from data_setup import dataloaders
from engine import train
from model import VanillaRNN
from utils import device

INPUT_SIZE = 5
OUTPUT_SIZE = 2
NUM_EPOCHS = 10
BATCH_SIZE = 16
HIDDEN_UNITS = 32
LEARNING_RATE = 0.01


train_dataloader, test_dataloader = dataloaders(
    train_path='data/train_data.csv',
    test_path='data/test_data.csv',
    batch_size=BATCH_SIZE
)

model = VanillaRNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_UNITS,
    output_size=OUTPUT_SIZE
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device
)