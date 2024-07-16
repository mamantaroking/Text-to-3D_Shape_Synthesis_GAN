import torch
from tqdm import tqdm
import os
from textto3dgan.preprocessing import textAndShapeLoader, dataset
from textto3dgan import utils, intro
from textto3dgan.intro import num_epochs, batch_size, dataroot, Dlr, Glr, plotter, plotting_FID, loss_Save, fid_Save
# from textto3dgan.new_texter_model7 import textD, textG, criterion, optimizerD, optimizerG, nz
from textto3dgan.new_texter_model8 import textD, textG, criterion, optimizerD, optimizerG, nz
# from textto3dgan.new_texter_model3 import textD, textG, criterion, optimizerD, optimizerG, nz
from SANITY_CHECK import ShowVoxelModel
from FIDType2 import calculate_fid
from torch.cuda.amp import GradScaler, autocast
import time
import csv
from torch.optim.lr_scheduler import StepLR

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Set device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize lists to keep track of progress
shape_list = []
G_losses = []
D_losses = []
fidPlot = []
accuracy_monitor = []
iters = 0
# fake_noise = torch.randn(batch_size, nz, 1, 1, 1, device=device)
fake_noise = torch.randn(batch_size, 1024, 1, 1, 1, device=device)
model_saved_path = 'model_save/'
image_saved_path = 'model_save/images/'
csv_file_path = 'model_save/images/losses.csv'
fid_file_path = 'model_save/images/fid.csv'


schedulerG = StepLR(optimizerG, step_size=1, gamma=1.2)
schedulerD = StepLR(optimizerD, step_size=2, gamma=0.5)

if __name__ == '__main__':
    def train(dataloader, textG, textD, criterion, optimizerG, schedulerD,schedulerG, num_epochs, device, iters=None):
        dset_len = {"train": len(dataset)}
        print(dset_len["train"])
        print("Dlr: ", Dlr)
        print("Glr: ", Glr)
        start = time.time()
        iters = 0  # Initialize 'iters' inside the function
        scaler = GradScaler()

        for epoch in range(num_epochs):


            for i, (data, text_embedding, labelling) in enumerate(tqdm(dataloader, 0)):  # Unpack the text embeddings and 3D shapes

                realData = data.to(device,non_blocking=True)
                text_embedding = text_embedding.to(device, non_blocking=True)  # Move the text embeddings to the device
                batches = realData.size(0)

                fakeData = textG(fake_noise, text_embedding)

                # ___Update Discriminator___
                # _real data with right text_
                textD.zero_grad()
                '''with autocast():
                    outputRealD = textD(realData, text_embedding).view(-1)
                    label = torch.full((batches,), 1, dtype=torch.float, device=device).float()
                    # print(outputRealD.type) # , label)
                    errDReal = criterion(outputRealD, label)'''
                label = torch.full((batches,), 1, dtype=torch.float, device=device).float()
                outputRealD = textD(realData, text_embedding).view(-1)
                errDReal = criterion(outputRealD, label)
                # scaler.scale(errDReal).backward()
                errDReal.backward()
                D_x = outputRealD.mean().item()


                # fake data
                '''with autocast():
                    outputFakeD = textD(fakeData.detach(), text_embedding.detach()).view(-1)
                    label.fill_(0)  # Fake labels are zero
                    errDFake = criterion(outputFakeD, label)'''
                outputFakeD = textD(fakeData.detach(), text_embedding.detach()).view(-1)
                label.fill_(0)  # Fake labels are zero
                errDFake = criterion(outputFakeD, label)
                # scaler.scale(errDFake).backward()
                errDFake.backward()
                D_G_z1 = outputFakeD.mean().item()
                errD = errDReal + errDFake
                # errD.backward()

                # scaler.step(optimizerD)
                # scaler.update()
                optimizerD.step()


                # ___Update Generator
                textG.zero_grad()
                label.fill_(1)  # We want the fake data to be seen as real
                '''with autocast():
                   outputG = textD(fakeData, text_embedding).view(-1)  # Pass the text embeddings to the discriminator
                    errG = criterion(outputG, label)'''
                outputG = textD(fakeData, text_embedding).view(-1)  # Pass the text embeddings to the discriminator
                errG = criterion(outputG, label)
                # scaler.scale(errG).backward()
                errG.backward()

                D_G_z3 = outputG.mean().item()
                # scaler.step(optimizerG)
                # scaler.update()
                optimizerG.step()

                if i % 20 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z3))
                iters += 1


            end = time.time()
            epoch_time = end - start
            # schedulerD.step()
            schedulerG.step()

            real_features = textD.main(realData).cpu().detach().reshape(96 * batch_size, 512).numpy()
            fake_features = textD.main(fakeData).cpu().detach().reshape(96 * batch_size, 512).numpy()
            fid = calculate_fid(real_features, fake_features)
            fidPlot.append(fid)
            print('FID Score:', fid)
            print("new Dlr", schedulerD.get_last_lr())
            print("new Glr", schedulerG.get_last_lr())

            print('Elapsed Time: {:.4} min'.format(epoch_time / 60.0))

            model_num = str(epoch + 1)

            if (epoch + 1) % utils.model_save_step == 0:
                print('model_saved, images_saved...')
                torch.save(textG.state_dict(), model_saved_path + '/G' + model_num + '.pth')
                torch.save(textD.state_dict(), model_saved_path + '/D.pth')
                D_losses.append(errD.item() / batch_size)
                G_losses.append(errG.item() / batch_size)

            if (epoch + 1) % 1 == 0:
                samples = fakeData.cpu().data[:16].squeeze().numpy()
                ShowVoxelModel(samples, image_saved_path, epoch)

        end_time = time.time()
        print(time.strftime('%H:%M:%S', time.localtime(start)))
        print(time.strftime('%H:%M:%S', time.localtime(end_time)))
        print('Total Time: {:.4} min'.format(end_time / 60.0))



    # Call the train function with the appropriate parameters
    train(textAndShapeLoader, textG, textD, criterion, optimizerG, schedulerD, schedulerG, num_epochs, device)
    plotter(D_losses, G_losses, image_saved_path)
    plotting_FID(fidPlot, image_saved_path)
    loss_Save(G_losses, D_losses, csv_file_path)
    fid_Save(fidPlot, fid_file_path)
    # accuring(accuracy_monitor, image_saved_path)
    # train(binvoxDataloader, netG, netD, criterion, optimizerG, optimizerD, num_epochs, device)
