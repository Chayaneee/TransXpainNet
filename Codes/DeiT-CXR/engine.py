import torch
import csv
from tqdm import tqdm



# training function
def train(model, dataloader, optimizer, criterion, train_data, device, scheduler):
    print('Training')
    model.cuda()
    model.train()
    counter = 0
    train_running_loss = 0.0
    
    
    

    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        counter += 1
        data, target = data['image'].cuda(), data['label'].cuda()
        optimizer.zero_grad()
        outputs = model(data)
        # apply sigmoid activation to get all the outputs between 0 and 1 if BCELoss is used as criterion
        #outputs = torch.sigmoid(outputs)
        # If BCEWithLogitsLoss is applied then skip the sigmoid activation of output
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        scheduler.step()  
            
    train_loss = train_running_loss / counter
    return train_loss

# validation function
def validate(model, dataloader, criterion, val_data, device):
    print('Validating')
    model.cuda()
    model.eval()
    counter = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            counter += 1
            data, target = data['image'].cuda(), data['label'].cuda()
            outputs = model(data)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs) 
            loss = criterion(outputs, target)
            val_running_loss += loss.item()
        
        val_loss = val_running_loss / counter
        return val_loss
