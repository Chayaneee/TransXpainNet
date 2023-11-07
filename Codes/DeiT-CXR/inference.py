import torch
import numpy as np
import models
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import *

# remove warnings
import warnings
warnings.filterwarnings("ignore")


from dataset import ImageDataset
from torch.utils.data import DataLoader

model_name = 'ViT'

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#intialize the model
model = models.model(model_name, pretrained=False, requires_grad=False).to(device)
# load the model checkpoint
checkpoint = torch.load('/mnt/data/chayan/CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0/results/Best_Model_BCE.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

src_dir = "/mnt/data/chayan/CheXpert/chexpertchestxrays-u20210408/"

train_csv = pd.read_csv('VisualCheXbert_classes.csv')
classes = train_csv.columns.values[5:19]    ### Genre

print(classes)
# prepare the test dataset and dataloader
test_data = ImageDataset(
   src_dir, train_csv, train=False, test=True
)
test_loader = DataLoader(
    test_data, 
    batch_size=1,
    shuffle=False
)




actual = []
predict = []

for counter, data in enumerate(test_loader):
    image, target = data['image'].to(device), data['label']
    
    actual.append(target.cpu().numpy())
    
    # get all the index positions where value == 1
    target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
    #print(target_indices)
    
    # get the predictions by passing the image through the model
    outputs = model(image)
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    max_value = torch.max(outputs)
    output = torch.where((outputs > 0.5) | (outputs == max_value), torch.ones_like(outputs), torch.zeros_like(outputs))
    output = output.detach().cpu()
    
    predict.append(output.numpy())
    
    #print(outputs)
    #print(output)
    #sorted_indices = np.argsort(output[0])
    best_indices = [i for i in range(len(output[0])) if output[0][i] ==1]
    #print(best_indices)
    #best = sorted_indices[-3:]
    string_predicted = ''
    string_actual = ''
    for i in range(len(best_indices)):
        string_predicted += f"{classes[best_indices[i]]}    "
    for i in range(len(target_indices)):
        string_actual += f"{classes[target_indices[i]]}    "
    image = image.squeeze(0)
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
    plt.savefig(f"/mnt/data/chayan/CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0/results/inference_{counter}.png")
    
actual = np.concatenate(actual)
predict = np.concatenate(predict)

print(actual)
print(predict)



## Define Overall average accuracy

def accuracy_multilabel(cm, num_class):
    
    accuracy = 0
    
    for i in range(num_class):
        true = np.diag(cm[i]).sum()
        total = cm[i].sum()
        acc = true / total
        accuracy += acc
    
    avg_acc = accuracy / num_class  
    
    return avg_acc
    


# # Confusion Matrix

cm = multilabel_confusion_matrix(actual, predict)
print(cm) 


avg_acc = accuracy_multilabel(cm, 14)


# ### classification_report

label_names = classes
report = classification_report(actual, predict, digits=3, target_names=label_names)

# # Print the report
print(report)


# ### Accuracy Metrics


auc =roc_auc_score(actual, predict) 
precision = precision_score(actual, predict, average='macro')
recall = recall_score(actual, predict, average='macro')
f1 = f1_score(actual, predict, average='macro')

# # Print the results
print("Overall_Accuracy:", avg_acc)
print("AUC:", auc)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
