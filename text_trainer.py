import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accuracy_classifier import calc_acc2
from preprocessing import textAndShapeLoader, dataset, accuracitator
import utils
# from textto3dgan.binvox_dataloader import binvoxDataloader, dataset
from intro import num_epochs, batch_size, dataroot, Dlr, Glr, plotter, accuring  # , nz
# om textto3dgan.newmodel import netD, netG, criterion, optimizerD, optimizerG
# from textto3dgan.text_model import textD,textG, criterion, optimizerD, optimizerG
from new_texter_model import textD,textG, criterion, optimizerD, optimizerG, nz
# from textto3dgan.utils import SavePloat_Voxels, generateZ
import datetime
from SANITY_CHECK import ShowVoxelModel
import time

# Set device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize lists to keep track of progress
shape_list = []
G_losses = []
D_losses = []
accuracy_monitor = []
iters = 0
fake_noise = torch.randn(batch_size, nz, 1, 1, 1, device=device)
model_saved_path = 'model_save/'
image_saved_path = 'model_save/images/'



if __name__ == '__main__':
    def train(dataloader, textG, textD, criterion, optimizerG, optimizerD, num_epochs, device, iters=None):
        dset_len = {"train": len(dataset)}
        print(dset_len["train"])
        print("Dlr: ")
        print(Dlr)
        print("Glr: ")
        print(Glr)
        iters = 0  # Initialize 'iters' inside the function
        for epoch in range(num_epochs):

            start = time.time()
            running_loss_G = 0.0
            running_loss_D = 0.0
            running_loss_adv_G = 0.0

            for i, (data, text_embedding, labelling) in enumerate(tqdm(dataloader, 0)):  # Unpack the text embeddings and 3D shapes

                realData = data.to(device)
                text_embedding = text_embedding.to(device)  # Move the text embeddings to the device
                # wrong_text_embedding = wrong_text_embedding.to(device)
                batches = realData.size(0)
                class_label = labelling.to(device)

                fakeData = textG(fake_noise, text_embedding)
                # fakeShapes = textG(fake_noise)
                # print(fakeData)
                label = torch.full((batches, ), 1, dtype=torch.float, device=device)
                # Z = generateZ(real_data)
                # d_real = netD(data)
                # fake = textG(Z, text_embedding)  # Pass the text embeddings to the generator
                # d_fake = netD(fake)
                # real_labels = torch.ones_like(realData).to(device)
                # d_real_loss = criterion(real_data, real_labels)

                # ___Update Discriminator___
                # _real data with right text_
                textD.zero_grad()
                # outputRealD = textD(realData, text_embedding).view(-1)
                outputRealD = textD(realData, text_embedding).view(-1)

                # label = torch.full((b_size,), 1, dtype=torch.float, device=device)
                # output = discriminator(real_data, text_embedding).view(-1)  # Pass the text embeddings to the discriminator
                # errDReal = criterion(outputRealD, torch.full_like(outputRealD,1,  device=device))
                errDReal = criterion(outputRealD, label)

                # errDReal = criterion(outputRealD, torch.full_like(outputRealD,1,  device=device))
                errDReal.backward()
                D_x = outputRealD.mean().item()

                ''''# _fake data with real text_
                outputFakeRightText = textD(fakeData.detach(), text_embedding.detach()).view(-1)
                label.fill_(0)
                errDFakeRightText = criterion(outputFakeRightText, label)
                errDFakeRightText.backward()
                D_G_z1 = outputFakeRightText.mean().item()

                # _Real data with wrong text_
                outputRealWrongText = textD(realData, wrong_text_embedding.detach()).view(-1)
                errDRealWrongText = criterion(outputRealWrongText, label)
                errDRealWrongText.backward()
                D_G_z2 = outputRealWrongText.mean().item()
                errD = errDReal + errDFakeRightText + errDRealWrongText
                optimizerD.step()
                D_G_z = D_G_z1 + D_G_z2'''

                # fake data
                outputFakeD = textD(fakeData.detach(), text_embedding.detach()).view(-1)
                # outputFakeD = outputFakeD[0]
                # outputFakeD = outputFakeD.detach()
                # fake_data = generator(noise, text_embedding)  # Pass the text embeddings to the generator
                label.fill_(0)  # Fake labels are zero
                # output = discriminator(fake_data.detach(), text_embedding).view(-1)  # Pass the text embeddings to the discriminator
                errDFake = criterion(outputFakeD, label)
                # errDFake = criterion(outputFakeD, torch.full_like(outputFakeD, 0, device=device))
                errDFake.backward()
                D_G_z1 = outputFakeD.mean().item()
                errD = errDReal + errDFake
                optimizerD.step()
                # fake_labels = torch.zeros_like(fake_data).to(device)

                # ___Update Generator
                textG.zero_grad()
                label.fill_(1)  # We want the fake data to be seen as real
                outputG = textD(fakeData, text_embedding).view(-1)  # Pass the text embeddings to the discriminator
                # outputG = outputG[0]
                # outputG = outputG.detach()
                errG = criterion(outputG, label)
                # errG = criterion(outputG, torch.full_like(outputG, 1, device=device))
                errG.backward()

                D_G_z3 = outputG.mean().item()
                optimizerG.step()

                if i % 80 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z3))
                iters += 1

            # epoch_loss_G = running_loss_G / dset_len["train"]
            # epoch_loss_D = running_loss_D / dset_len["train"]
            # epoch_loss_adv_G = running_loss_adv_G / dset_len["train"]

            end = time.time()
            epoch_time = end - start
            total_time =+ end
            acc_samples = fakeData.cpu().data[:77].detach()

            acc_forwarder = accuracitator(acc_samples, class_label)
            acc_dataloader = DataLoader(acc_forwarder, batch_size=1)

            # print('Epochs-{} ({}) , D(x) : {:.4}, D(G(x)) : {:.4}'.format(epoch, "train", epoch_loss_D,
                                                                              # epoch_loss_adv_G))
            print('Elapsed Time: {:.4} min'.format(epoch_time / 60.0))

            if (epoch + 1) % utils.model_save_step == 0:
                print('model_saved, images_saved...')
                torch.save(textG.state_dict(), model_saved_path + '/G.pth')
                torch.save(textD.state_dict(), model_saved_path + '/D.pth')
                # acc_samples = fakeData.cpu().data[:77].detach().squeeze(0)
                # print(acc_samples.shape)
                # print(class_label)
                # print(calc_acc)
                # acc_additor = (acc_samples, class_label)
                # calc_acc(acc_samples, class_label)
                accuracy = calc_acc2(acc_dataloader)
                D_losses.append(errD.item() / batch_size)
                G_losses.append(errG.item() / batch_size)
                accuracy_monitor.append(accuracy) #  / batch_size)

            if (epoch + 1) % 10 == 0:
                # samples = fakeShapes.cpu().data[:8].squeeze().numpy()
                samples = fakeData.cpu().data[:16].squeeze().numpy()
                # print (samples.shape)
                # image_saved_path = '../images'

                ShowVoxelModel(samples, image_saved_path, epoch)
                # G_losses.append(errG.item())
                # D_losses.append(errD.item())

        end_time = time.time()
        print(time.strftime('%H:%M:%S', time.localtime(
            end_time)))
        # print('Total Time: {:.4} min'.format(end_time / 60.0))

    # Call the train function with the appropriate parameters
    train(textAndShapeLoader, textG, textD, criterion, optimizerG, optimizerD, num_epochs, device)
    plotter(D_losses, G_losses, image_saved_path)
    accuring(accuracy_monitor, image_saved_path)
    # train(binvoxDataloader, netG, netD, criterion, optimizerG, optimizerD, num_epochs, device)
