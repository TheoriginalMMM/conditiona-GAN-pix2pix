import torch
from datasets import ComicsDataset,MapDataset,ViewDataset
import conf
from torchvision.utils import save_image
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import generator

import imageio
from PIL import Image,ImageDraw, ImageFont
import re

# To save exempels
 
def save_some_examples(gen, val_loader, epoch):
    x, y = next(iter(val_loader))
    x, y = x.to(conf.DEVICE), y.to(conf.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#

        eval_directory = conf.EVAL_DIR + "/online/"+conf.DATA_NAME
        if not os.path.exists(eval_directory):
            os.makedirs(eval_directory)

        
        save_image(y_fake* 0.5 + 0.5, eval_directory+f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, eval_directory+ f"/input_{epoch}.png")
        
        if epoch == 1:
            save_image(y * 0.5 + 0.5, conf.EVAL_DIR + "/"+conf.DATA_NAME+ f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=conf.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def matplotlib_imshow(img1, img2):
    n_im = img1.shape[0]
    npimg1 = img1.detach().cpu().numpy()
    npimg2 = img2.detach().cpu().numpy()
    list1 = [npimg1[i] for i in range(n_im)]
    list2 = [npimg2[i] for i in range(n_im)]
    figure, _ = disp(list1+list2,(2,n_im),1)
    figure.tight_layout()
    return figure


def weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        #tch.nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


# display image
def disp(im_list, shape=(1,1), scale=1):
    fig, ax = plt.subplots(nrows=shape[0], ncols=shape[1], squeeze=False)
    fig.set_size_inches(5*scale*shape[1]/2.54,5*scale*shape[0]/2.54)
    
    if shape[0]==shape[1]==1:
        im_list = [im_list]
    if len(im_list)>shape[0]*shape[1]:
        raise ValueError('The product of figure shape must be'+
                         ' lower than im_list length')
    
    k = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            ax[i,j].imshow(im_list[k], cmap='gray');
            ax[i,j].xaxis.set_visible(False)
            ax[i,j].yaxis.set_visible(False)
            k += 1
    return fig, ax



def generate_train_gif(directory="./"+conf.EVAL_DIR + "/online/"+conf.DATA_NAME+"/",speed=0.1):
    images = []

    input_files = [file for file in sorted(os.listdir(directory)) if file.startswith("input_")]
    print(input_files[:10])
    indexs_ = [int(re.findall(r'\d+', f)[0]) for f in input_files]
    indexs_.sort()
    
    #for i in range(len(input_files)):
    for i in indexs_:
        text = f"Epoch : {i}"
        input_file_path = os.path.join(directory, f"input_{i}.png")
        output_file_path = os.path.join(directory, f"y_gen_{i}.png")
        
        i_image = Image.open(input_file_path)
        o_image = Image.open(output_file_path)
        concatenated_image = Image.fromarray(np.concatenate((i_image, o_image), axis=1))

        text_image = Image.new(concatenated_image.mode, (concatenated_image.width, 30), (255, 255, 255))

        draw = ImageDraw.Draw(text_image)
        font = ImageFont.truetype("arial.ttf", 20)
        text_width, text_height = draw.textsize(text, font)
        draw.text(((text_image.width - text_width) / 2, 0), text, font=font, fill=(0, 0, 0))

        concatenated_image.paste(text_image, (0,0))
        concatenated_image.save(directory+text+".png", "PNG")
        images.append(concatenated_image)

    print("saving ! in :" ,directory + 'train.gif')
    imageio.mimsave(directory + conf.DATA_NAME+'_train.gif', images, duration=speed)


# TODO : reffactor data loader 
def test_generator_offline(weights_path,val_directory,data_set,nb_tests_to_be_ploted):

    # Creating generator 
    gen = generator.Generator(in_channels=3, features=64).to(conf.DEVICE)
    print(weights_path)
    
    # Loading weights     
    checkpoint = torch.load(weights_path, map_location=conf.DEVICE)
    gen.load_state_dict(checkpoint["state_dict"])

    if data_set == "ComicsDataset":
        val_dataset = ComicsDataset(root_dir=val_directory+"/validation/",from_faces=True)
        resutl_path = val_directory+"/results/"
    elif data_set == "MapDataset":
        val_dataset = MapDataset(val_directory+"/val/")
        resutl_path = val_directory+"/results_arial_map/"
    elif data_set == "ViewDatast":
        val_dataset = ViewDataset(val_directory+"/val/")
        resutl_path = val_directory+"/results_map_arial/"
    else:
        pass
        
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    loop = tqdm(val_loader, leave=True)
    images = []
    l1_losses = []

    for idx, (x, y) in enumerate(loop):

        # Geting x and y
        x = x.to(conf.DEVICE)*0.5+0.5
        y = y.to(conf.DEVICE)*0.5+0.5
        y_gen = gen(x)*0.5+0.5
        errors_map = torch.abs(y_gen-y)

        l1 = torch.sum(errors_map)

        #transpose = lambda x : np.transpose(x, (0, 2, 3, 1))
        #print((transpose(x).shape))
        #result = Image.fromarray(np.concatenate((transpose(x), transpose(y_gen),transpose(y),transpose(errors_map)), axis=1))
        concatenation_array = torch.cat((x,y_gen,y,errors_map), axis=-1)
        save_image(concatenation_array, resutl_path+"concatenation_"+str(idx)+".png")
        
        l1_losses.append(l1.cpu().detach())
        
    l1_losses_arg_sort = np.array(l1_losses).argsort()
    top_nb_test_indexs = l1_losses_arg_sort[:nb_tests_to_be_ploted]
    worst_nb_test_indexs = l1_losses_arg_sort[-nb_tests_to_be_ploted:]
    
    print(top_nb_test_indexs,worst_nb_test_indexs)
    get_image_paths = lambda x : f"{resutl_path}concatenation_{x}.png"
    best_paths = [get_image_paths(x) for x in top_nb_test_indexs ]
    worst_paths = [get_image_paths(x) for x in worst_nb_test_indexs ]
    
    best_realisations_image = stack_images_vertically(best_paths)
    worst_realisations_image = stack_images_vertically(worst_paths)
    
    best_realisations_image.save(resutl_path+f"best_of_{nb_tests_to_be_ploted}"+".png", "PNG")
    print(resutl_path+f"best_of_{nb_tests_to_be_ploted}"+".png")
    worst_realisations_image.save(resutl_path+f"worst_of_{nb_tests_to_be_ploted}"+".png", "PNG")



def stack_images_vertically(image_paths):
    images = [Image.open(path) for path in image_paths]
    widths, heights = zip(*(i.size for i in images))
    total_height = sum(heights)
    max_width = max(widths)
    result = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        result.paste(im, (0, y_offset))
        y_offset += im.size[1]

    return result
if __name__ == "__main__":
    print("HELLO!")
    # generate_train_gif(speed=0.3)
    #test_generator_offline("weights/map.view.gen.pth.tar","data/maps",4)
    #test_generator_offline("weights/map.view.gen.pth.tar","data/maps","MapDataset",4)
    list_paths_ = os.listdir("/home/mmm/Documents/MMA/PAAI/fails_map_arial")
    list_paths = ["/home/mmm/Documents/MMA/PAAI/fails_map_arial/"+x for x in list_paths_[:-1]]
    image_ = stack_images_vertically(list_paths)
    image_.save(f"/home/mmm/Documents/MMA/PAAI/fails_map_arial/combinaison_fails_map_arial.png","PNG")