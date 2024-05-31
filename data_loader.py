
import os
import cv2


def load_annot_image_level(path):
    path = path + 'annotations.txt'
    video_annot =[]    ### using it as list looks better
    video_annot_dic ={}

    with open(path, 'r') as annot_file:
        annot_file = annot_file.read().splitlines()
        
        for image in annot_file:
            
            image_annot = image.split(' ')
            image_annot = image_annot[:2]
            video_annot.append(image_annot)
            video_annot_dic[image_annot[0]] = image_annot[1]
    
    return video_annot, video_annot_dic


def load_image(video_path, visualize = False):
    images_num = os.listdir(video_path)
    images_num.sort()
    ## clip is a list of images each image express the all clip  (ex. in video #0/ clip #13361/ image #a13361)
    video_images = [video_path + image_num + '/' + image_num +'.jpg' for image_num in images_num if image_num.isdigit()]


    if visualize:
        for i in range(len(video_images)):
            visualize_image(video_images[i])
    ##########################################
    ### here i can use map function to visualize the video_images instead of using for loop
    ##########################################

    return video_images


def save_annot(path, output_path):
    video_dirs = os.listdir(path)
    video_dirs.sort()
    video_dirs = [os.path.join(path, video_dir) for video_dir in video_dirs if video_dir.isdigit()]
    for i, video_dir in enumerate(video_dirs):
        str_i = video_dir.split('/')
        str_i = str_i[-1]
        output_file = output_path + str_i + "/"
        # output_file = os.path.join(output_path, str_i)
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        output_file = output_file + "imageLevelAnnot.txt"
        
        video_dir = video_dir + '/'
        video_annot, video_annot_dic = load_annot_image_level(video_dir)

        with open(output_file, 'w') as output:
            for image in video_annot:
                output.write(image[0] + ',' + image[1] + '\n')
                

def visualize_image(image_path):
    
    image = cv2.imread(image_path)
    cv2.imshow("Image", image)
    cv2.waitKey(100)


if __name__ == "__main__":
    path = '/home/husammm/Desktop/Courses/ML/Projects/GroupActivityRecognation/Data/videos/'
    output_path = '/home/husammm/Desktop/Courses/ML/Projects/GroupActivityRecognation/Data/features/imageLevel/annotations/'
    # print(os.listdir(path))
    # video_annot, video_annot_dic = load_annot_image_level(path = path)
    # load_image(path, visualize = True)
    save_annot(path, output_path)


