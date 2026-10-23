import torch
from torch import nn
import random
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from helper_functions import accuracy_fn


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
BATCH_SIZE = 32


# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)

image, label = train_data[0]

class_names = train_data.classes

#print(f"Image shape: {image.shape}")
#plt.imshow(image.squeeze())

# Plot more images
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, lbl = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze())
    plt.title(class_names[lbl])
    plt.axis(False)

train_dataloader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)


train_features_batch, train_labels_batch = next(iter(train_dataloader))

random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()

img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis("Off");

flatten_model = nn.Flatten()

x = train_features_batch[0]
output = flatten_model(x)

#create a convolutional neural network
class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape: int,
                 hidden_shape: int,
                 output_shape: int,):
        super().__init__()

        self.block_1 = nn.Sequential(
           nn.Conv2d(
              in_channels=input_shape,
              out_channels=hidden_shape,
              kernel_size=3,
              stride=1,
              padding=1),
           nn.ReLU(),
           nn.Conv2d(
              in_channels=hidden_shape,
              out_channels=hidden_shape,
              kernel_size=3,
              stride=1,
              padding=1),
            nn.ReLU(),
            nn.MaxPool2d(
               kernel_size=2,
               stride=2)
        )

        self.block_2 = nn.Sequential(
           nn.Conv2d(
              hidden_shape,
              hidden_shape,
              3,
              padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_shape,
                      hidden_shape,
                      3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
           nn.Flatten(),
           nn.Linear(in_features=hidden_shape*7*7,
                     out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
       x = self.block_1(x)
       x = self.block_2(x)
       x = self.classifier(x)
       return x

model_02 = FashionMNISTModelV2(input_shape=1,
                            hidden_shape=10,
                            output_shape=len(class_names)).to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_02.parameters(), lr=0.01)

def train_step(model:torch.nn.Module,
               data_loader:torch.utils.data.dataloader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
   
   train_loss, train_acc = 0, 0

   model.to(device)
   model.train()

   for batch, (X, y) in enumerate(data_loader):
      X, y = X.to(device), y.to(device)

      y_pred = model(X)
      loss = loss_fn(y_pred, y)

      train_loss += loss.item()
      train_acc += accuracy_fn(y_true = y, y_pred = y_pred.argmax(1))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

   train_loss /= len(data_loader)
   train_acc /= len(data_loader)

   return train_loss, train_acc

def test_step(model:torch.nn.Module,
              data_loader:torch.utils.data.dataloader,
              loss_fn:torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):

   test_loss, test_acc = 0, 0
   model.to(device)
   model.eval()

   with torch.inference_mode():
      for X, y in data_loader:

         X, y = X.to(device), y.to(device)

         test_pred = model(X)
         test_loss += loss_fn(test_pred, y).item()
         test_acc += accuracy_fn(y_true=y,
                                 y_pred=test_pred.argmax(1))

      test_loss /= len(data_loader)
      test_acc /= len(data_loader)

   return test_loss, test_acc 

best_test_acc = 0.0
history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
epochs = 10
for epoch in range(epochs):

   print(f"\nEpoch: {epoch + 1} / {epochs}")

   train_loss, train_acc = train_step(
      model=model_02,
      data_loader=train_dataloader,
      loss_fn=loss_fn,
      optimizer=optimizer,
      accuracy_fn=accuracy_fn,
      device=device
   )

   test_loss, test_acc = test_step(
      model=model_02,
      data_loader=test_dataloader,
      loss_fn=loss_fn,
      accuracy_fn=accuracy_fn,
      device=device
   )

   history["train_loss"].append(train_loss)
   history["test_loss"].append(test_loss)
   history["train_acc"].append(train_acc)
   history["test_acc"].append(test_acc)

   if test_acc > best_test_acc:
      best_test_acc = test_acc
      torch.save(model_02.state_dict(), "best_fashion_mnist.pth")

   print(
           f"Train loss: {train_loss:.4f} | "
           f"Train accuracy: {train_acc:.2f}% | "
           f"Test loss: {test_loss:.4f} | "
           f"Test accuracy: {test_acc:.2f}%"
       )

print(f"\nBest test accuracy: {best_test_acc:.2f}%")

def plot_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for ax, key, title in [(ax1, "loss", "Loss"), (ax2, "acc", "Accuracy (%)")]:
        ax.plot(epochs, history[f"train_{key}"], label="train")
        ax.plot(epochs, history[f"test_{key}"], label="test")
        ax.set(title=title, xlabel="Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_curves(history)

def make_prediction(model: torch.nn.Module, data):
    
    model.eval()
    pred_probs = []
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)

test_samples = list()
test_lables = list()
for sample, lable in random.sample(list(test_data), k=9):
   test_samples.append(sample)
   test_lables.append(lable)

pred_probs = make_prediction(model=model_02,
                             data=test_samples)
pred_classes = pred_probs.argmax(dim=1)

plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):

   plt.subplot(nrows, ncols, i+1)
   plt.imshow(sample.squeeze())

   pred_lable = class_names[pred_classes[i]]
   truth_lable = class_names[test_lables[i]]
   title_text = f"Pred: {pred_lable} | Truth: {truth_lable}"

   if pred_lable == truth_lable:
      plt.title(title_text, fontsize=10, c="g")
   else:
      plt.title(title_text, fontsize=10, c="r") 
   plt.axis(False)
plt.show()


