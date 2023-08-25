import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.datasets as ds
from torch.utils.data import DataLoader as dl
from torchvision.transforms import transforms as tforms
from torch.utils.tensorboard import SummaryWriter as sw
from DCGAN_gen_disc import Discriminator, Generator, initialize_weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 32
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = tforms.Compose(
    [
        tforms.Resize(IMAGE_SIZE),
        tforms.ToTensor(),
        tforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        )
    ]
)

dataset = ds.MNIST(root="dataset/DCGAN", train=True, transform= transforms,
                   download=True)

loader = dl(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(Z_DIM, FEATURES_GEN, CHANNELS_IMG).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn((32, Z_DIM, 1, 1)).to(device)

writer_real = sw(f"logs/real")
writer_fake = sw(f"logs/fake")

step = 0
gen.train()
disc.train()

total_data_size=len(loader)
print(gen)
print(disc)
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        print(f"iteration: {batch_idx}")
        real = real.to(device)
        #print(f'Real Shape: {real.shape}')
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
        #print(f'Noise Shape: {noise.shape}')
        fake = gen(noise)
        #print(f'Fake Shape: {fake.shape}')
        ###Train Discriminator :
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like((disc_real)))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        disc_loss = (loss_disc_fake + loss_disc_real)/2
        disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()


        ###Train Generator
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()

        opt_gen.step()


        writer_real.add_scalar("loss gen", loss_gen.item(), total_data_size*epoch+batch_idx)
        writer_real.add_scalar("loss disc", disc_loss.item(), total_data_size * epoch + batch_idx)

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] \ "
                f"Loss Disc: {disc_loss:.4f}, Loss Gen: {loss_gen:.4f}"
            )
        if batch_idx%10==0:
            with torch.no_grad():
                #fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                fake = gen(fixed_noise)
                #data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

        step += 1