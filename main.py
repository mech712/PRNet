import os.path as path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import skimage.io as sio
import skimage.transform as st

import api
import utils.write as uw

prn = api.PRN(is_dlib=True)

#img_path = "./TestImages/AFLW2000/m1.jpg"
img_path = input("Введите абсолютный или относительный путь к фотографии >> ")
abspath = path.abspath(img_path)

image = sio.imread(abspath)

h, w, c = image.shape
if c>3:
    image = image[:,:,:3]

max_size = max(image.shape[0], image.shape[1])
if max_size>1000:
    image = st.rescale(image, 1000./max_size)
    image = (image*255).astype(np.uint8)
pos = prn.process(image) # use dlib to detect face


vertices = prn.get_vertices(pos)
colors = prn.get_colors(image, vertices)


save_dir = input("Введите абсолютный или относительный путь к директории, в которой будет сохранен файл >> ")
img_name = path.basename(abspath)
save_path = path.join(path.abspath(save_dir), img_name+".obj")

uw.write_obj_with_colors(save_path, vertices, prn.triangles, colors) #save 3d face(can open with meshlab)
