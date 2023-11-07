import models
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from engine import train, validate
from dataset import ImageDataset
from torch.utils.data import DataLoader
from utils import SaveBestModel, save_model, save_latest_model, resume
from torchsummary import summary
import csv

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR

matplotlib.style.use('ggplot')
# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# remove warnings
import warnings
warnings.filterwarnings("ignore")

import wandb



model_name = 'ViT' # ViT/ResNet

#intialize the model
model = models.model(model_name, pretrained=True, requires_grad=True)

# Resume with latest model
resume(model, resume = False)

model.cuda()



print(summary(model, input_size =(3,384,384)))

print('Start Training')

# learning parameters
lr = 0.01
max_lr = 0.1
epochs = 20
batch_size = 8
optimizer = optim.Adamax(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()
save_best_model = SaveBestModel()



#lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

wandb.init( project= "DeiT__new",
            config = {"Learning_rate": "lr",
                       "Epochs": "epochs",
                  }
          )

####### read the training csv file

src_dir = "/mnt/data/chayan/CheXpert/chexpertchestxrays-u20210408/"

train_csv = pd.read_csv('train_visualCheXbert_classes.csv')
# train dataset
train_data = ImageDataset(
   src_dir, train_csv, train=True, test=False
)
# validation dataset
valid_data = ImageDataset(
   src_dir, train_csv, train=False, test=False
)
# train data loader
train_loader = DataLoader(
    train_data, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
 #   prefetch_factor=8,
 #   pin_memory=True
)


num_training_steps = len(train_loader) * epochs 
num_warmup_steps = int(num_training_steps * 0.1)

scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps = num_warmup_steps,
                num_training_steps = num_training_steps,
                #num_cycles = 1, # for cosine
                lr_end = 1e-7,
                power = 2,
                last_epoch = -1
                )




##steps_per_epoch = int(len(train_data)/batch_size)
#lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps=None, epochs=epochs, steps_per_epoch=steps_per_epoch)

# validation data loader
valid_loader = DataLoader(
    valid_data, 
    batch_size=batch_size,
    shuffle=False,
    num_workers=8
    #prefetch_factor=8,
    #pin_memory=True
)



##### start the training and validation
train_loss = []
valid_loss = []

LR = []

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    train_epoch_loss = train(
        model, train_loader, optimizer, criterion, train_data, device, scheduler
    )
    valid_epoch_loss = validate(
        model, valid_loader, criterion, valid_data, device
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    wandb.log({"Train_Loss":train_epoch_loss, "Val_Loss": valid_epoch_loss})

    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {valid_epoch_loss:.4f}')

    save_best_model(
        valid_epoch_loss, epoch, model, optimizer, criterion,
    )
    
    save_latest_model(epoch, model, optimizer, criterion)
        
    
    #lr_scheduler.step()
    curr_lr = optimizer.param_groups[0]['lr']
    LR.append(curr_lr)
    
    print("Current learning rate: {}".format(optimizer.param_groups[0]['lr']))
    
 
# Specify the file path to save the CSV file
csv_file_name = 'learning_rates.csv'

# Save the learning rates to the CSV file
with open(csv_file_name, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Training Step', 'Learning Rate'])  # Write header row
    for step, lr in enumerate(LR):
        writer.writerow([step, lr])   
        
save_model(epochs, model, optimizer, criterion)    

# plot and save the train and validation line graphs
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/mnt/data/chayan/CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0/results/loss_curve.png')
plt.figure()