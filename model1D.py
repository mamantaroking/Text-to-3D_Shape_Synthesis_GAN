import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from params import ngpu, ngf, nc, ndf, Glr, Dlr, beta1, batch_size
from preprocessing import device
from torchinfo import summary
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
nz = 256

# initialize custom weights for 'netG' and 'netD'
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


'''Define Generator'''


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        # text embedding layers
        self.text_embedding = nn.Sequential(
            nn.Linear(768, nz * 4, bias=False),  # 768 is the size of BERT embeddings,
            nn.BatchNorm1d(nz * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz * 4, nz * 2, bias=False),  # 768 is the size of BERT embeddings,
            nn.BatchNorm1d(nz * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(nz * 2, nz, bias=False),  # 768 is the size of BERT embeddings,
            # nn.BatchNorm1d(nz),
            # nn.LeakyReLU(0.2, inplace=True)
        )

        # main generator architecture
        self.main = nn.Sequential(
            # nn.ConvTranspose3d(nz + 768, ngf * 8, 4, 1, 0, bias=False),  # Adjust the number of input channels
            # input is random noise Z, going into a convolution
            nn.ConvTranspose3d(nz * 4, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            # state size. ''(ngf*8) x 4 x 4''
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            # state size. ''(ngf*4) x 8 x 8''
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            # state size. ''(ngf*2) x 16 x 16''
            nn.ConvTranspose3d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 1),
            nn.ReLU(True),
            # state size. ''(ngf) x 32 x 32''
            nn.ConvTranspose3d(ngf * 1, nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            nn.Sigmoid(),
            nn.BatchNorm3d(1)
            # state size. ''(nc) x 64 x 64''
        )


    def forward(self, noise, text):
        text = text.squeeze(1)
        text = self.text_embedding(text)
        text = text.view(text.shape[0],  text.shape[1], 1, 1, 1)
        z = torch.cat([text, noise], 1)
        return self.main(z)

    '''def forward(self, noise, text):
        print("GENERATOR")
        print("noise: ")
        print(noise.shape)
        print("text: ")
        print(text.shape)
        text = text.squeeze(1)
        print("text after squeeze: ")
        print(text.shape)
        text = self.text_embedding(text)
        print("text after model: ")
        print(text.shape)
        text = text.view(text.shape[0],  text.shape[1], 1, 1, 1)
        print("text after view: ")
        print(text.shape)
        z = torch.cat([text, noise], 1)
        print("after concatenate: ")
        print(z.shape)
        return self.main(z)'''


'''Instantiate the generator'''
# Create the generator
textG = Generator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    textG = nn.DataParallel(textG, list(range(ngpu)))

# Apply the ''weights_init'' function to randomly initialize all weights
# to ''mean=0'', ''stdev=0.02''.
textG.apply(weights_init)

# Print the model
# print(textG)
noise = [64, 512, 1, 1, 1]
text = [64, 768]  # assuming the text input is a 1024-dimensional vector
summary(model=textG, input_size=[noise, text])


'''Define Discriminator'''
# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        # self.fc = nn.Linear(17 * 17 * 17 + 768, 17 * 17 * 17)  # 768 is the size of BERT embeddings
        self.main = nn.Sequential(
            # input is ''(nc) x 64 x 64''
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ''(ndf) x 32 x 32''
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ''(ndf*2) x 16 x 16''
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ''(ndf * 4) x 8 x 8''
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ''(ndf*8) x 4 x 4''
            # nn.Conv3d(ndf * 8, ndf * 16, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

        self.embedding = nn.Sequential(
            nn.Linear(768, nz * 2, bias=False),  # 768 is the size of BERT embeddings,
            nn.BatchNorm1d(nz * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(nz * 4, nz * 2, bias=False),
            # nn.BatchNorm1d(nz * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz * 2, nz, bias=False),
            nn.BatchNorm1d(nz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz, nz * 2, bias=False),
            nn.BatchNorm1d(nz * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.output = nn.Sequential(
            nn.Conv3d(nz * 4, nz * 2, 1, 1, 0, bias=False),
            nn.BatchNorm3d(nz * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nz * 2, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # nn.BatchNorm3d(1)
        )


    def forward(self, x, text):
        x_out = self.main(x)  # Extract features from the input using the discriminator architecture
        text = text.squeeze(1)
        text_out = self.embedding(text)  # Apply text embedding
        text_out = text_out.repeat(4, 4, 4, 1, 1).permute(3, 4, 0, 1, 2)
        out = torch.cat([text_out, x_out], dim=1)  # Concatenate the input features and the text embedding
        out = self.output(out)  # Final discriminator output
        return out

    '''def forward(self, x, text):
        print("DISCRIMINATOR")
        print("3d: ")
        print(x.shape)
        print("text: ")
        print(text.shape)
        x_out = self.main(x)  # Extract features from the input using the discriminator architecture
        print("3d after model: ")
        print(x_out.shape)
        text = text.squeeze(1)
        print("text after squeeze: ")
        print(text.shape)
        text_out = self.embedding(text)  # Apply text embedding
        print("text after model: ")
        print(text_out.shape)
        text_out = text_out.repeat(4, 4, 4, 1, 1).permute(3, 4, 0, 1, 2)
        print(text_out.shape)
        print("text after view: ")
        print(text_out.shape)
        out = torch.cat([text_out, x_out], dim=1)  # Concatenate the input features and the text embedding
        print("after concatenate: ")
        print(out.shape)
        out = self.output(out)  # Final discriminator output
        print("final output: ")
        print(out.shape)
        return out #, x_out'''


'''Instantiate the Discriminator'''
# Create the discriminator
textD = Discriminator(ngpu).to(device)

# Handle multi-GPU if desires
if (device.type == 'cuda') and (ngpu > 1):
    textD = nn.DataParallel(textD, list(range(ngpu)))

# Apply the ''weights_init'' function to randomly initialize all weights
# like this: ''to mean=0, stdev=0.2''
textD.apply(weights_init)

# Print the model
# print(netD)
# summary(model=netD, input_size=(1, 1, 64, 64, 64))

'''Define the loss functions'''
# Initialize the ''BCELoss'' function
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()

def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)

    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)
    return gen_loss

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    return crit_loss


def get_gradient(crit, real, fake, text, epsilon):
    mixed_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = crit(mixed_images, text)

    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,

    )[0]
    return gradient

# Create batch of latent vectors that we will use to visualize
# the progression of the generator
# fixed_noise = torch.randn(64, nz, 1, 1, 1, device=device)

# Establish convention for real and fake laebls during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
# optimizerD = optim.Adam(textD.parameters(), lr=Dlr, betas=(beta1, 0.999))
# optimizerG = optim.Adam(textG.parameters(), lr=Glr, betas=(beta1, 0.999))
optimizerD = optim.SGD(textD.parameters(), lr=Dlr)
optimizerG = optim.SGD(textG.parameters(), lr=Glr)
# optimizerD = optim.RMSprop(textD.parameters(), lr=Dlr)
# optimizerG = optim.RMSprop(textG.parameters(), lr=Glr)

noise = [64, 1, 64, 64, 64]
text = [64, 768]  # assuming the text input is a 1024-dimensional vector
summary(model=textD, input_size=[noise, text])
# summary(model=textD.main, input_size=noise)