python train.py\
 --model "efficientnet_b2"\
 --epoch 200\
 --num-classes 2\
 --data "/home/nas/Research_Group/Personal/Andrew/birth_event_detection/dataset/train_and_val"\
 --batch-size 64\
 --device 2\
 --imgsz 224\
 --patience 15

python val.py\
 --weights "/home/ubuntu/Classification-Validation-Tools/first_model/weight.pth"\
 --data "/home/nas/Research_Group/Personal/Andrew/birth_event_detection/dataset/test"\
 --batch-size 64\
 --device 2\
 --imgsz 224\
 --name "cm_test_dataset"