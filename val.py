"""
Quick start:

python val.py --weights "/home/ubuntu/Classification-Validation-Tools/weight.pth" --data "/home/nas/Research_Group/Personal/Andrew/birth_event_detection/dataset/train_and_val/val" --batch-size 64 --device 2 --imgsz 224 --name "cm_validation_dataset"

I hold your back bro.

Ben,
June 18th, 2023
""" 

import argparse
import torch
import torchvision
from torchvision.models import efficientnet_b2

from utils import data_transform, predict, ImageFolderWithPaths, plot_matrix
from sklearn.metrics import confusion_matrix, accuracy_score

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None, help='initial weights path')
    parser.add_argument('--data', type=str, default=None, help='dataset directory path')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--device', type=int, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--name', type=str, default='confusion_matrix_validation', help='name of images')

    opt = parser.parse_args()
    return opt

def main(opt):
    weights = opt.get('weights')
    data_dir = opt.get('data')
    batch_size = opt.get('batch_size')
    device = opt.get('device')
    image_size = opt.get('imgsz')
    model_name = opt.get('name')

    """Detect if we have a GPU available"""
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    """Load model"""
    if weights.endswith('.tar'):
        checkpoint = torch.load(weights)
        model = efficientnet_b2()
        model.load_state_dict(checkpoint['state_dict'], strict=False) # trun .pyh.tar into readilbe
    else:
        model = torch.load(weights)

    """Load testing data"""
    data_transforms = data_transform(image_size)
    test_dataset = ImageFolderWithPaths(f"{data_dir}", data_transforms["val"])    

    """Predict"""
    preds, trues, paths = predict(test_dataset, model, batch_size, device)
    with open('result.txt', 'w') as txt:
        txt.write('pred, true, path')
        for pred, true, path in zip(preds, trues, paths):
            txt.write(f'\n{pred}, {true}, {path}')
    
    """Using predicted results to calculate an accuracy score and draw a confusion matrix"""
    acc_score = accuracy_score(trues, preds)
    cm = confusion_matrix(trues, preds)
    nor_cm = confusion_matrix(trues, preds, normalize="true")
    print(f'Accuracy : {acc_score}')
    print(f'Confusion Matrix :\n {cm}')

    new_val_classes = test_dataset.classes
    plot_matrix(nor_cm, new_val_classes, model_name)

    print("I'm so handsome.")


if __name__ == "__main__":
    opt = parse_opt()
    opt = vars(opt)
    
    for key in opt.keys():
        if opt.get(key) is None:
            value = input(f"Please input {key} values: ")

            if key in ["device", "batch_size"]:
                value = int(value)

            opt[key] = value

    for key, value in opt.items():
        print(f"{key}: {value}")
    main(opt)