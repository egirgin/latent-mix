import pickle

import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                                                    
                                                    
                                                    
with open("./augmix_imgs.pickle", "rb") as adv_file:
    adv_samples = pickle.load(adv_file)      
    
    
with open("./augmix_labels.pickle", "rb") as label_file:
    labels = pickle.load(label_file)

adv_set = [(adv_samples[i], labels[i]) for i in range(len(labels))]


dataloader = torch.utils.data.DataLoader(adv_set, batch_size=4, shuffle=True)



import torchvision.models as models
net = models.resnet18().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # clear the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        # calculate loss
        loss = criterion(outputs, labels)
        
        # backprop 
        loss.backward()

        # update weights
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:    # print every 2000 mini-batches
            print("Epochs: {}, Iteration: {}, Loss: {}".format(epoch+1, i, loss.item()))
        #running_loss = 0.0

    per_class_acc(net, dataloader, classes)

print('Finished Training')


PATH = './lsun_augmix_resnet18.pth'
torch.save(net.state_dict(), PATH)
