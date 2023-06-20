"""
Quick start:

python train.py --model "efficientnet_b2" --epoch 200 --num-classes 2 --data "/home/nas/Research_Group/Personal/Andrew/modelTraining/train_and_val" --batch-size 64 --device 2 --imgsz 224

I hold your back bro.

Ben,
June 18th, 2023
""" 
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
from pathlib import Path

from utils import *
from sklearn.metrics import confusion_matrix, accuracy_score

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help="model's name")
    parser.add_argument('--epoch', type=int, default=None, help='epochs')
    parser.add_argument('--lr', type=int, default=0.005, help='learning rate')
    parser.add_argument('--momentum', type=int, default=0.9, help='momentum')
    parser.add_argument('--num-classes', type=int, default=None, help='num')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--patience', type=int, default=50, help='eraly stop ')
    parser.add_argument('--device', type=int, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--data', type=str, default=None, help='dataset directory path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='inference size (pixels)')

    opt = parser.parse_args()
    return opt

def main(opt):
    model_name = opt.get('model')
    num_epochs = opt.get('epoch')
    lr = opt.get('lr')
    momentum = opt.get('momentum')
    num_classes = opt.get('num_classes')
    data_dir = opt.get('data')
    batch_size = opt.get('batch_size')
    patience = opt.get('patience')
    device = opt.get('device')
    input_size = opt.get('imgsz')
    feature_extract = None
    
    os.makedirs(os.path.join(Path().resolve(), 'run'), exist_ok=True)

    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    """Initialize the model for this run"""
    model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    """
    Data augmentation and normalization for training
    Just normalization for validation  
    """  
    data_transforms = data_transform(input_size)
    
    print("Initializing Datasets and Dataloaders...")    

    """Create training and validation datasets"""
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    """Create training and validation dataloaders"""
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=16) for x in ['train', 'val']}

    """Send the model to GPU"""
    model_ft = model_ft.to(device)

    """    
    Gather the parameters to be optimized/updated in this run. If we are
    finetuning we will be updating all parameters. However, if we are
    doing feature extract method, we will only update the parameters
    that we have just initialized, i.e. the parameters with requires_grad
    is True.
    """
    params_to_update = model_ft.parameters()

    print("Params to learn:")
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    """Observe that all parameters are being optimized"""
    optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=momentum)
    """Setup the loss fxn"""
    criterion = nn.CrossEntropyLoss()

    """Train and evaluate"""    
    model_ft, val_acc_hist, val_loss_hist, train_acc_hist, train_loss_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_epochs, is_inception=(model_name=="inception"), patience=patience)
    torch.save(model_ft, f'/run/{model_name}.pth')

    plot_val_train_hist(num_epochs, val_loss_hist, train_loss_hist, model_name, 'Loss')
    plot_val_train_hist(num_epochs, val_acc_hist, train_acc_hist, model_name, 'Accuracy')

    pred, true, paths = predict(image_datasets['val'], model_ft, batch_size, device)
    print(paths)

    """Using predicted results to calculate an accuracy score and draw a confusion matrix"""
    acc_score = accuracy_score(true, pred)
    cm = confusion_matrix(true, pred)
    nor_cm = confusion_matrix(true, pred, normalize="true")
    print(f'Accuracy : {acc_score}')
    print(f'Confusion Matrix :\n {cm}')

    new_val_classes = image_datasets['val'].classes
    print(new_val_classes)
    plot_matrix(nor_cm, new_val_classes, 'confusion_matrix')

    print("I'm so handsome.")


if __name__ == "__main__":
    opt = parse_opt()
    opt = vars(opt)
    
    for key in opt.keys():
        if opt.get(key) is None:
            value = input(f"Please input {key} values: ")

            if key in ["device", "batch_size", "epoch", "num_classes"]:
                value = int(value)

            opt[key] = value

    for key, value in opt.items():
        print(f"{key}: {value}")

    main(opt)