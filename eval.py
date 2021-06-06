import pickle
import argparse

import torch
import torchvision
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--adv", action="store_true")
parser.add_argument("--inv", action="store_true")
parser.add_argument("--augmix", action="store_true")
parser.add_argument("--comp", action="store_true")


args = parser.parse_args()

classes = ["church", "living_room"]
def per_class_acc(net, dataloader, classes):

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                    accuracy))

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Resize((256, 256))] )

church = torchvision.datasets.LSUN(root=".", classes=['church_outdoor_val'], transform=transform)
living_room = torchvision.datasets.LSUN(root=".", classes=['living_room_val'], transform=transform)


dataset = []
for i in range(len(church)):
    dataset.append(church[i])
    dataset.append((living_room[i][0], 1))

del church, living_room

dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
del dataset


PATH = './lsun_resnet18.pth'
if args.adv:
    PATH = './lsun_adv_resnet18.pth'
if args.inv:
    PATH = './lsun_inv_resnet18.pth'
if args.augmix:
    PATH = './lsun_augmix_resnet18.pth'
if args.comp:
    PATH = './lsun_comp_resnet18.pth'


net = models.resnet18().to(device)
net.load_state_dict(torch.load(PATH))

net.eval()
print("Clean Acc:")
per_class_acc(net, dataloader, classes)


with open("./pgd.pickle", "rb") as adv_file:
    adv_samples = pickle.load(adv_file)      
    
    
with open("./labels.pickle", "rb") as label_file:
    labels = pickle.load(label_file)

adv_set = [(adv_samples[i], labels[i]) for i in range(len(labels))]


adv_loader = torch.utils.data.DataLoader(adv_set, batch_size=4, shuffle=True)

print("Adversarial Acc:")
per_class_acc(net, adv_loader, classes)

