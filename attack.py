import pickle

import torch
import torchvision
import torchvision.models as models

import foolbox as fb
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

imgs = torch.cat([item[0].to(device).unsqueeze(0) for item in dataset], dim=0)
labels = torch.tensor([item[1] for item in dataset]).to(device)

del dataset


PATH = './lsun_resnet18.pth'


net = models.resnet18().to(device)
net.load_state_dict(torch.load(PATH))

net.eval()


fmodel = fb.PyTorchModel(net, bounds=(0, 1))
"""
clean_acc = accuracy(fmodel, imgs, labels)
print(f"clean accuracy:  {clean_acc * 100:.1f} %")
"""

attack = fb.attacks.LinfPGD()
#attack = fb.attacks.LinfFastGradientAttack()

epsilons = [0.01]
raw_advs, clipped_advs, success = attack(fmodel, imgs[:300], labels[:300], epsilons=epsilons)


print(sum(success[0]).item()/300)


with open("./pgd.pickle", "bw+") as picklefile:
    pickle.dump(clipped_advs[0], picklefile)
    
    
with open("./pgd_labels.pickle", "bw+") as picklefile:
    pickle.dump(labels[:300], picklefile)

