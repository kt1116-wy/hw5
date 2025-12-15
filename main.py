# main.py
import torch
from torch.utils.data import DataLoader, random_split

RANDOM_STATE = 2025
dataset = HW5Dataset('train_dataset/meta.csv')
random_seed_generator = torch.Generator().manual_seed(RANDOM_STATE)

# TODO (5P): Create training and validation dataset based on `dataset`
# and `random_seed_generator`. You should split the dataset into 80%
# samples for the training, and 20% samples for the validation. You
# should use `random_seed_generator` to ensure consistency of the
# result.
# CHECK: https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=random_seed_generator
)

# TODO (5P): Create training and validation data loader based on
# `train_dataset` and `val_dataset`. For val_loader, please set
# `shuffle=False` for the consistency of the result.
# CHECK: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders
# e.g. train_loader = DataLoader(..., collate_fn=collate_batch)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_batch
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_batch
)


# Optional: change different hyperparameters, to get the best model.
hidden_size = 64
num_layers = 3
lr = 5e-4
patience = 8
max_n_epochs = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = HW5Model(hidden_size=hidden_size, num_layers=num_layers, lr=lr, device=device)
model.train_epochs(train_loader, val_loader, patience=patience, max_n_epochs=max_n_epochs)
