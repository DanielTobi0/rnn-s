import torch
import data_setup, engine, model, utils


INPUT_SIZE = 5
OUTPUT_SIZE = 2
NUM_EPOCHS = 5
BATCH_SIZE = 16
HIDDEN_UNITS = 64
LEARNING_RATE = 0.01


train_dataloader, test_dataloader = data_setup.dataloaders(
    train_path='data/train_data.csv',
    test_path='data/test_data.csv',
    batch_size=BATCH_SIZE
)

model = model.VanillaRNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_UNITS,
    output_size=OUTPUT_SIZE
).to(utils.device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=utils.device
)