import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples,weights_init,matplotlib_imshow
import torch.nn as nn
import torch.optim as optim
import conf
from datasets import MapDataset,ViewDataset,ComicsDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,writer,epoch
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(conf.DEVICE)
        y = y.to(conf.DEVICE)
        #################################################################
        ##################### Train Discriminator #######################
        with torch.cuda.amp.autocast():

            # Generator inference => y_fake
            y_fake = gen(x)

            # Discriminator inference on the real data (y)
            D_real = disc(x, y)

            # Discriminator loss on the  real data
            D_real_loss = bce(D_real, torch.ones_like(D_real))

            # Descriminator inference on the generated data
            D_fake = disc(x, y_fake.detach())

            # Descriminator loss on the generated data 
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            
            # TOTAL LOSS of the desctiminator 
            # Total loss : Descriminator to train slower
            # Devide the loss by 2 to make the descriminator learn slowlyf
            D_loss = (D_real_loss + D_fake_loss) /2


        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        ###################### END Discriminator training ########################## 
        #############################################################################

        ############################################################################
        ###################### Train generator #####################################

        with torch.cuda.amp.autocast():
            
            # Descriminator inference on the generated or fake  data
            D_fake = disc(x, y_fake)

            # Trying to trik the discriminator => Ones like 
            ############## Generator loss  
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))

            ##### Calculate the  L1 loss 
            L1 = l1_loss(y_fake, y) * conf.L1_LAMBDA

            ## Generator loss 
            G_loss = G_fake_loss + L1


        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        ######################## END GENERATOR TRAINING  ###########################
        #############################################################################


        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

        if epoch % 10 == 0:
            # ...log the running loss for each epoch
            writer.add_scalar('Loss D',
                                D_loss,
                                epoch)

            writer.add_scalar('Loss G',
                                G_fake_loss,
                                epoch)

            writer.add_scalar('Loss L1',
                                L1/conf.L1_LAMBDA,
                                epoch)

            writer.add_scalar('G total loss ',
                                G_loss,
                                epoch)
            D_real=torch.sigmoid(D_real).mean().item(),
            D_fake=torch.sigmoid(D_fake).mean().item(),
            writer.add_scalar('proba real',
                                torch.tensor(D_real),
                                epoch)
            writer.add_scalar('proba fake',
                                torch.tensor(D_fake),
                                epoch)


def main():
    # Tensor bord summary 
    writer = SummaryWriter()

    # Creating discriminator 
    disc = Discriminator(in_channels=3).to(conf.DEVICE)
    # Creating generator 
    gen = Generator(in_channels=3, features=64).to(conf.DEVICE)

    # Init models weights !
    disc.apply(weights_init)
    gen.apply(weights_init)



    # Optimizers 
    opt_disc = optim.Adam(disc.parameters(), lr=conf.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=conf.LEARNING_RATE, betas=(0.5, 0.999))
    
    # BCE with LogitsLoss // There are another options but this what works 
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()


    if conf.LOAD_MODEL:
        load_checkpoint(
            conf.CHECKPOINT_GEN, gen, opt_gen, conf.LEARNING_RATE,
        )
        load_checkpoint(
            conf.CHECKPOINT_DISC, disc, opt_disc, conf.LEARNING_RATE,
        )

    train_dataset = ComicsDataset(conf.DATA_FOLDER,from_faces=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=conf.BATCH_SIZE,
        shuffle=True,
        num_workers=conf.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    val_dataset = ComicsDataset(root_dir="data/face2comics/validation/",from_faces=True)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(conf.NUM_EPOCHS):
        # Train function 
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,writer,epoch
        )

        if conf.SAVE_MODEL and epoch % 100 == 0:
            save_checkpoint(gen, opt_gen, filename=f"face.comic.{conf.CHECKPOINT_GEN}")
            save_checkpoint(disc, opt_disc, filename=f"face.comic.{conf.CHECKPOINT_DISC}")



        if epoch%10==0:
            # Send to tensor board instead of this ! 
            save_some_examples(gen, val_loader, epoch)


if __name__ == "__main__":
    main()