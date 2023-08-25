import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as ds
from torch.utils.data import DataLoader as dl
from torchvision.transforms import transforms as tforms
from torch.utils.tensorboard import SummaryWriter as sw

class Discriminator(nn.Module):
  def __init__(self, img_dims):
    super().__init__()
    self.disc = nn.Sequential(
        nn.Linear(img_dims, 128),
        nn.LeakyReLU(0.1),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
  def forward(self, x):
    return self.disc(x)


class Generator(nn.Module):
  def __init__(self, img_dims, z_dim):
    super().__init__()
    self.gen = nn.Sequential(
        nn.Linear(z_dim, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, img_dims) ,
        nn.Tanh(),
    )
  def forward(self, x):
      return self.gen(x)

#Hyperparameters etc
def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    z_dim = 64
    img_dim = 28*28*1

    batch_size = 32
    num_epochs = 52

    disc = Discriminator(img_dim).to(device)
    gen = Generator(img_dim, z_dim).to(device)

    fixed_noise = torch.randn((batch_size, z_dim)).to(device)

    transforms = tforms.Compose( [tforms.ToTensor(), tforms.Normalize((0.5,), (0.5,))]
                                )

    dataset = ds.MNIST(root="dataset/", transform = transforms, download = True)

    loader = dl(dataset= dataset, batch_size = batch_size, shuffle = True)

    optimizer_disc = optim.Adam(disc.parameters(), lr = lr)
    optimizer_gen = optim.Adam(gen.parameters(), lr = lr)

    criterion = nn.BCELoss()

    writer_fake= sw(f"run/GAN_MNIST/fake")
    writer_real = sw(f"run/GAN_MNIST/real")
    step = 0

    for epoch in range(num_epochs):
      for batch_idx, (real, _) in  enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]


        ###Train Discriminator : maximize log(D(real)) + log (1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)

        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_real))
        lossD = (lossD_real + lossD_fake)/2
        disc.zero_grad()
        lossD.backward()
        optimizer_disc.step()

        ###Train Generator : minimize log (1-D(G(z))) --> This expression leads to saturating gradients or sort of weak gradients, so actually maximize log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        optimizer_gen.step()



        ###Tensorboard code

        if batch_idx == 0:
          print(
              f"Epoch [{epoch}/{num_epochs}] \ "
              f"Loss Disc: {lossD:.4f}, Loss Gen: {lossG:.4f}"
          )
        with torch.no_grad():
          fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
          data = real.reshape(-1, 1, 28, 28)
          img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
          img_grid_real = torchvision.utils.make_grid(data, normalize=True)

          writer_fake.add_image(
              "Mnist Fake Images", img_grid_fake, global_step = step
          )

          writer_real.add_image(
              "Mnist real Images", img_grid_real, global_step=step
          )

        step += 1

if __name__ == '__main__':
    run()