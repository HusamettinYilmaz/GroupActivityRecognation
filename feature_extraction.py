import torch
import os
import numpy as np

import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models

from PIL import Image
from data_loader import load_image


def check_cuda():
    ## check torch version
    print(f"torch version is:{torch.__version__}")

    ## check cuda avilability (GPU)
    if torch.cuda.is_available():
        print("Cuda is avilable")

        ## check number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs:{num_gpus}")

        ## Details of each cuda device
        for i in range(num_gpus):
            print(f"Device {i+1}: {torch.cuda.get_device_name(i)}")

    else:
        ("No cuda 'GPU' found, Use CPU")


def preprocess_data(image_level = True):
    if image_level:
        ## preprocess the image with some transformations
        pre_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        ## since they are cropped images no need to recrop them
        pre_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return pre_transforms


def prepare_model():
    ## check kind of device GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## select model to use
    model = models.resnet50(pretrained=True)

    # Remove the classification head (i.e., the fully connected layers)
    model = nn.Sequential(*(list(model.children())[:-1]))
    # model = nn.Sequential(*(list(model.children())[:-1]))
    ## Send the model to the device (CPU or GPU)
    model.to(device)
    
    ## Set model to the evaluation mode
    model.eval()

    return model, device


def extract_features(path, output_root):
    ## prepare pretransforms and and model
    pre_transforms = preprocess_data(image_level= True)
    model, device = prepare_model()
        
    video_dirs = os.listdir(path)
    video_dirs.sort()
    ## Load images
    # video_paths = [path + str(i) + '/' for i in range(55)]  ## video_paths = [.0/, .1/, .2/ and etc]
    video_paths = [os.path.join(path, video_dir + '/') for video_dir in video_dirs if video_dir.isdigit()]

    for i, video_path in enumerate(video_paths):
        str_i = video_path.split('/')
        str_i = str_i[-2]

        ## video_images is a list of image's paths each image path express the all clip  (ex. in video #0/ clip #13361/ image #a13361)
        video_images = load_image(video_path, visualize = False)
        
        num_images = os.listdir(video_path)
        num_images.sort()

        output_files = os.path.join(output_root, str_i) + '/'
        if not os.path.exists(output_files):
            os.makedirs(output_files)
        
        ## With torch no grad
        with torch.no_grad():
            for i, image_path in enumerate(video_images):
                image = Image.open(image_path).convert('RGB')
                preprocessed_image = pre_transforms(image).unsqueeze(0)
                ## data to cuda
                preprocessed_image = preprocessed_image.to(device)
                ##############
                dnn_repr = model(preprocessed_image)
                dnn_repr = dnn_repr.view(1, -1)

                output_file = output_files + num_images[i]

                # Save feature vector using GPU
                if device.type == 'cuda':
                    torch.save(dnn_repr, output_file)
                
                # Save feature vector using CPU in numpy data type
                else:
                    # Move feature vector back to CPU for saving
                    feature_vector = dnn_repr.cpu().numpy()

                    # Save feature vector using CPU
                    np.save(output_file, feature_vector)
            

            
        


if __name__ == "__main__":

    path = '/home/husammm/Desktop/Courses/ML/Projects/GroupActivityRecognation/Data/videos/'
    output_root = '/home/husammm/Desktop/Courses/ML/Projects/GroupActivityRecognation/Data/features/imageLevel/resnet/'
    check_cuda()
    extract_features(path = path, output_root = output_root)

    
    
