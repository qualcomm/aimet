# pylint: disable=all
# setup
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from aimet_torch.batch_norm_fold import fold_all_batch_norms

# General setup that can be changed as needed
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = torchvision.models.mobilenet_v2(pretrained=True).eval().to(device)

batch_size = 64
PATH_TO_IMAGENET = ...
data = torchvision.datasets.ImageNet(PATH_TO_IMAGENET, split="train")
data_loader = DataLoader(data, batch_size=batch_size)

dummy_input = torch.randn(1, 3, 224, 224).to(device)
fold_all_batch_norms(model, dummy_input.shape)

# Callback function to pass calibration data through the model
def forward_pass(model: torch.nn.Module, batches):
    with torch.no_grad():
        for batch, (images, _) in enumerate(data_loader):
            images = images.to(device)
            model(images)
            if batch >= batches:
                break

# Basic ImageNet evaluation function
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, labels in tqdm(data_loader):
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            correct += (logits.argmax(1) == labels).type(torch.float).sum().item()
    accuracy = correct / len(data_loader.dataset)
    return accuracy

# step_1
from aimet_torch.v2.quantsim import QuantizationSimModel, QuantScheme
sim = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.training_range_learning_with_tf_init)

calibration_batches = 10
sim.compute_encodings(forward_pass, calibration_batches)

accuracy = evaluate(sim.model, data_loader)
print(f"PTQ model accuracy: {accuracy}")
# step_2
# Training loop can be replaced with any custom training loop
def train(model, data_loader, optimizer, loss_fn):
    model.train()
    for data, labels in tqdm(data_loader):
        data, labels = data.to(device), labels.to(device)
        logits = model(data)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(sim.model.parameters(), lr=1e-5)

epochs = 2
for epoch in range(epochs):
    train(sim.model, data_loader, optimizer, loss_fn)
# step_3
accuracy = evaluate(sim.model, data_loader)
print(f"Model accuracy after QAT: {accuracy}")
# step_4
sim.export(path="./", filename_prefix="quantized_mobilenetv2", dummy_input=dummy_input.cpu())