from facenet_pytorch import MTCNN
import torch
import os
import cv2
from tqdm import tqdm
workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(
    image_size=256, margin=0, min_face_size=50,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

if __name__ == '__main__':
    # ---------------------------------<for video>--------------------------------#
    # 原始数据集路径（根据自身路径修改）
    load_path = '/home/kangcaixin/chenjiawei/videos'
    save_path = '/home/kangcaixin/chenjiawei/physical_256'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for dir_num in tqdm(os.listdir(load_path)):
        dir_name = os.path.join(load_path, dir_num)
        save_name = os.path.join(save_path, dir_num)
        if not os.path.exists(save_name):
            os.mkdir(save_name)
        #
        print('开始对%s对象进行截取' % (dir_name))

        for data_n, iter in zip(os.listdir(dir_name), range(len(os.listdir(dir_name)))):
            data_n_split = data_n.split('.')
            save_data_n_split = data_n_split[0]
            save_name_2 = os.path.join(save_name, save_data_n_split)
            if not os.path.exists(save_name_2):
                os.mkdir(save_name_2)
            data_video = os.path.join(dir_name, data_n)
            cap = cv2.VideoCapture(data_video)
            c = 1
            frameRate = 1  # 帧数截取间隔（每隔15帧截取一帧）

            while (True):
                ret, frame = cap.read()
                if ret:
                    if (c % frameRate == 0):
                        #frame = cv2.flip(frame, -1)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_tensor, prob = mtcnn(frame,
                                                  save_path=save_name_2 + '/' + str(c) + ".jpg",
                                                  return_prob=True)
                        # align(image=frame, save_path=save_name_2 + '/' + str(data_num) + '_' + str(c) + ".jpg")
                        # 这里是将截取的图像，进行裁剪保存在本地
                    c += 1
                    # cv2.waitKey(0)
                else:
                    # print("所有帧都已经保存完成")
                    break
            cap.release()

        print('%s已完成' % (dir_name))