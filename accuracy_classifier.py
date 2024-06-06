import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from tqdm import tqdm

from textto3dgan import utils
from textto3dgan.classifier_dataloader import dataloader, device
from textto3dgan.intro import ngpu

# classes = ('chair', 'cone', 'die', 'doughnut', 'shoes', 'soccer_ball')
classes = ('chair', 'shoes')
# torch.use_deterministic_algorithms(True, warn_only=True)
torch.use_deterministic_algorithms(False)

class Net(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.conv1 = nn.Conv3d(1, 64, 4, 2, 1, bias=False)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(64, 128, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv3d(128, 256, 4, 2, 1, bias=False)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 6)

    '''def forward(self, x):
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv3(x)))
        print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        x = F.relu(self.fc3(x))
        print(x.shape)
        x = F.relu(self.fc4(x))
        print(x.shape)
        x = self.fc5(x)
        print(x.shape)
        return x'''

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


classifier = Net(ngpu).to(device)
# print(classifier)
shape = [64, 1, 64, 64, 64]
# summary(model=classifier, input_size=[shape])


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
model_saved_path = 'C:/Users/Maman/PycharmProjects/fyp/textto3dgan/classifer_saves/'

'''if __name__ == '__main__':
    for epoch in range(50):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(tqdm(dataloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs , labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % len(dataloader) == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(dataloader):.3f}')
                # running_loss = 0.0

            if (epoch + 1) % utils.model_save_step == 0:
                torch.save(classifier.state_dict(), model_saved_path + '/classification.pth')

    print('Finished Training')'''

PATH = 'C:/Users/Maman/PycharmProjects/fyp/textto3dgan/classifer_saves/classification.pth'
classifier = Net(ngpu).to(device)
classifier.load_state_dict(torch.load(PATH))
classifier.eval()


def accurator(dataloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            # first_batch = next(iter(dataloader))
            # print(first_batch)
            shapes, labels = data
            # shapes = data
            # labels = data
            # print(shapes)
            # print(labels)
            shapes = shapes.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = classifier(shapes)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(1)))
            # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(1)))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct // total

        print(f'Accuracy of the network on the dataset {100 * correct // total} %')
        # print(f'(Accuracy {100 * correct // total} %)')


def accurator2(dataloader):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            # first_batch = next(iter(dataloader))
            # print(first_batch)
            shapes, labels = data
            # shapes = data
            # labels = data
            # print(shapes)
            # print(labels)
            shapes = shapes.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = classifier(shapes)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
            print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(1)))
            # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(1)))

        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


# if __name__ == '__main__':
def calc_acc(dataloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            first_batch = next(iter(dataloader))
            # print(first_batch)
            shapes, labels = data
            # shapes = data
            # labels = data
            # print(shapes)
            # print(labels)
            # shapes = sample.to(device)
            shapes = shapes.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = classifier(shapes)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(14)))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total

        # print(f'Accuracy of the network on the dataset {100 * correct // total} %')
        # print(f'(Accuracy {100 * correct // total} %)')
        print(f'(Accuracy {accuracy} %)')
        return accuracy

def calc_acc2(dataloader, threshold=0.9999):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            first_batch = next(iter(dataloader))
            # print(first_batch)
            shapes, labels = data
            # shapes = data
            # labels = data
            # print(shapes)
            # print(labels)
            # shapes = sample.to(device)
            shapes = shapes.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = classifier(shapes)
            # Get the probabilities using softmax
            probabilities = F.softmax(outputs, dim=1)
            # Get the predicted class with the highest probability
            _, predicted = torch.max(probabilities, 1)
            # Compare the highest probability with the threshold
            confident_predictions = probabilities.max(dim=1).values > threshold
            # Consider only the predictions that are above the threshold
            total += confident_predictions.sum().item()
            correct += (predicted[confident_predictions] == labels[confident_predictions]).sum().item()

        # print(f'Accuracy of the network on the dataset {100 * correct // total} %')
        # print(f'(Accuracy {100 * correct // total} %)')
        accuracy = 100 * correct / total if total > 0 else 0
        print(f'(Accuracy {accuracy} %)')
        return accuracy

# accurator(dataloader)
# accurator2(dataloader)
# calc_acc(dataloader)
calc_acc2(dataloader)