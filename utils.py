import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResnetBlock(nn.Module):
    def __init__(self, input_channels, num_channels):
        super(ResnetBlock, self).__init__()
        self.bn = nn.BatchNorm2d(num_channels)
        self.conv1 = nn.Conv2d(
            input_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x):
        out = F.relu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))
        out = torch.cat((out, x), 1)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.resnet1 = ResnetBlock(128, 128)
        self.resnet2 = ResnetBlock(256, 128)
        self.resnet3 = ResnetBlock(384, 128)
        self.resnet4 = ResnetBlock(512, 128)
        self.resnet = nn.Sequential(
            self.resnet1, self.resnet2, self.resnet3, self.resnet4
        )
        self.conv3 = nn.ConvTranspose2d(
            640, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(
            64, 3, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn4 = nn.BatchNorm2d(3)

    def forward(self, input_image):
        out = F.relu(self.bn1(self.conv1(input_image)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.resnet(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.tanh(self.bn4(self.conv4(out)))
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, first_input):
        out = F.leaky_relu(self.conv1(first_input), 0.2)
        out = F.leaky_relu(self.bn1(self.conv2(out)), 0.2)
        out = F.leaky_relu(self.bn2(self.conv3(out)), 0.2)
        out = F.sigmoid(self.conv4(out))
        return out


class Joint_Model(nn.Module):
    def __init__(self, generator1, generator2, discriminator):
        super(Joint_Model, self).__init__()
        self.generator1 = generator1
        self.generator2 = generator2
        self.discriminator = discriminator

    def forward(self, first_input, second_input):

        for param in self.generator1.parameters():
            param.requires_grad = True

        for param in self.discriminator.parameters():
            param.requires_grad = False

        for param in self.generator2.parameters():
            param.requires_grad = False

        gen1_out_first = self.generator1(first_input)
        discriminator_output = self.discriminator(gen1_out_first)
        gen1_out_second = self.generator1(second_input)
        gen2_out_gen1 = self.generator2(gen1_out_first)
        gen2_out_second = self.generator2(second_input)
        gen1_out_gen2 = self.generator1(gen2_out_second)

        return discriminator_output, gen1_out_second, gen2_out_gen1, gen1_out_gen2


def retrieve_real(dataset, batch_size, mid_shape):
    random_index = np.random.randint(0, dataset.shape[0], batch_size)
    image_x = dataset[random_index]
    image_y = torch.ones((batch_size, 1, mid_shape, mid_shape)).cuda()
    return image_x, image_y


def generate_fake(x_real, generator, mid_shape):
    image_x = generator(x_real).detach()
    image_y = torch.zeros((len(image_x), 1, mid_shape, mid_shape)).cuda()
    return image_x, image_y


def update_pool(pool, images, limit=8):
    out = []
    for image in images:
        if len(pool) < limit:
            pool.append(image)
            out.append(image)
        elif np.random.random() < 0.5:
            out.append(image)
        else:
            index = np.random.randint(0, limit)
            out.append(pool[index])
            pool[index] = image
    return torch.stack(out)


def train(
    generator_a,
    generator_b,
    discriminator_a,
    discriminator_b,
    joint_a,
    joint_b,
    dataset,
    batch_size,
    epochs,
):

    # Defining the loss functions and optimizers
    discriminator_loss = nn.BCELoss()
    generator_loss = nn.L1Loss()
    joint_a_optimizer = torch.optim.Adam(
        joint_a.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    joint_b_optimizer = torch.optim.Adam(
        joint_b.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    discriminator_a_optimizer = torch.optim.Adam(
        discriminator_a.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    discriminator_b_optimizer = torch.optim.Adam(
        discriminator_b.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )

    # Initialize the image pools
    image_pool_a, image_pool_b = [], []
    d_out_dim = 17

    # Move the models to the GPU
    discriminator_a = discriminator_a.cuda()
    discriminator_b = discriminator_b.cuda()
    generator_a = generator_a.cuda()
    generator_b = generator_b.cuda()
    joint_a = joint_a.cuda()
    joint_b = joint_b.cuda()

    discriminator_a = discriminator_a.train()
    discriminator_b = discriminator_b.train()
    generator_a = generator_a.train()
    generator_b = generator_b.train()
    joint_a = joint_a.train()
    joint_b = joint_b.train()

    for i in range(epochs):
        for t, (train_a, train_b) in enumerate(dataset):
            (train_a, train_b) = (train_a.cuda(), train_b.cuda())

            x_real_a, y_real_a = retrieve_real(train_a, batch_size, d_out_dim)
            x_real_b, y_real_b = retrieve_real(train_b, batch_size, d_out_dim)

            x_fake_a, y_fake_a = generate_fake(x_real_a, generator_a, d_out_dim)
            x_fake_b, y_fake_b = generate_fake(x_real_b, generator_b, d_out_dim)

            x_fake_a = update_pool(image_pool_a, x_fake_a)
            x_fake_b = update_pool(image_pool_b, x_fake_b)

            # Descriminiator A parameter update
            discriminator_a_optimizer.zero_grad()
            for param in discriminator_a.parameters():
                param.requires_grad = True
            discriminator_a_output_real = discriminator_a(x_real_a)
            discriminator_a_output_fake = discriminator_a(x_fake_a)
            discriminator_a_loss_real = discriminator_loss(
                discriminator_a_output_real, y_real_a
            )
            discriminator_a_loss_fake = discriminator_loss(
                discriminator_a_output_fake, y_fake_a
            )

            discriminator_global_loss_a = (
                discriminator_a_loss_real + discriminator_a_loss_fake
            )
            discriminator_a_loss_real.backward()
            discriminator_a_loss_fake.backward()
            discriminator_a_optimizer.step()

            # Discriminator B parameter update
            discriminator_b_optimizer.zero_grad()
            for param in discriminator_b.parameters():
                param.requires_grad = True
            discriminator_b_output_real = discriminator_b(x_real_b)
            discriminator_b_output_fake = discriminator_b(x_fake_b)
            discriminator_b_loss_real = discriminator_loss(
                discriminator_b_output_real, y_real_b
            )
            discriminator_b_loss_fake = discriminator_loss(
                discriminator_b_output_fake, y_fake_b
            )

            discriminator_global_loss_b = (
                discriminator_b_loss_real + discriminator_b_loss_fake
            )

            discriminator_b_loss_real.backward()
            discriminator_b_loss_fake.backward()
            discriminator_b_optimizer.step()

            # Joint A parameter update
            joint_a_optimizer.zero_grad()
            (
                discriminator_output_a,
                gen1_out_second_a,
                gen2_out_gen1_a,
                gen1_out_gen2_a,
            ) = joint_a(x_real_b, x_real_a)
            joint_a_discriminator_loss = discriminator_loss(
                discriminator_output_a, y_real_a
            )
            joint_a_generator1_loss = generator_loss(gen1_out_second_a, x_real_a)
            joint_a_generator1_loss.backward()
            joint_a_generator2_loss = generator_loss(gen2_out_gen1_a, x_real_b)
            joint_a_generator3_loss = generator_loss(gen1_out_gen2_a, x_real_a)
            joint_a_generator3_loss.backward()

            joint_a_loss = (
                joint_a_discriminator_loss
                + 5 * joint_a_generator1_loss
                + 5 * joint_a_generator2_loss
                + 10 * joint_a_generator3_loss
            )
            joint_a_optimizer.step()

            # Joint B parameter update
            joint_b_optimizer.zero_grad()
            (
                discriminator_output_b,
                gen1_out_second_b,
                gen2_out_gen1_b,
                gen1_out_gen2_b,
            ) = joint_b(x_real_a, x_real_b)
            joint_b_discriminator_loss = discriminator_loss(
                discriminator_output_b, y_real_b
            )
            joint_b_generator1_loss = generator_loss(gen1_out_second_b, x_real_b)
            joint_b_generator1_loss.backward()
            joint_b_generator2_loss = generator_loss(gen2_out_gen1_b, x_real_a)
            joint_b_generator2_loss
            joint_b_generator3_loss = generator_loss(gen1_out_gen2_b, x_real_b)
            joint_b_generator3_loss.backward()

            joint_b_loss = (
                joint_b_discriminator_loss
                + 5 * joint_b_generator1_loss
                + 5 * joint_b_generator2_loss
                + 10 * joint_b_generator3_loss
            )
            joint_b_optimizer.step()

            if t % 100 == 0:
                print(
                    f"Epoch {i}, Batch {t}, Discriminator A Loss: {discriminator_global_loss_a}, Discriminator B Loss: {discriminator_global_loss_b}, Joint A Loss: {joint_a_loss}, Joint B Loss: {joint_b_loss}"
                )

        # Save the models at every epoch
        torch.save(joint_a.state_dict(), f"joint_a.pt")
        torch.save(joint_b.state_dict(), f"joint_b.pt")
        torch.save(discriminator_a.state_dict(), f"discriminator_a.pt")
        torch.save(discriminator_b.state_dict(), f"discriminator_b.pt")
