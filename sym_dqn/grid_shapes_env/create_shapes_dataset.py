import os
import imageio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from skimage.transform import resize

folder = 'dataset'
os.mkdir(folder)
dataset_csv = open(os.path.join(folder,'dataset.csv'), 'w+')
dataset_csv.write('class,path\n')

data_size = 1 #per type
pixel_size = (10,10)

settings = [
{ 'background_color': c[0],
  'entity_colors': {'agent':c[1],'cross':c[1],'circle':c[1],'square':c[1]} }
for c in [['white','black'],['black','white'],['red','blue'],['blue','red']]
]

def unique_file(basename, ext):
    actualname = "%s%s" % (basename, ext)
    c = 1
    while os.path.exists(actualname):
        actualname = "%s_%d%s" % (basename, c, ext)
        c+=1
    return actualname

""" Setting masks """
masks = {}
for entity_type in ['agent','cross','circle','square']:
    f = os.path.join(os.path.dirname(__file__),'images','{}.png'.format(entity_type))
    mask = imageio.imread(f)
    mask = resize(mask, pixel_size, mode='edge', preserve_range=True)
    masks[entity_type] = np.tile(mask[...,3:], (1,1,3)) / 255.

for setting in settings:
    """ Setting colors """
    background_color = to_rgb(setting['background_color'])
    background_color = np.array(background_color)[None, None, :]

    entity_colors = {}
    for entity_type in ['agent','cross','circle','square']:
        entity_color = to_rgb(setting['entity_colors'][entity_type])
        entity_colors[entity_type] = np.array(entity_color)[None, None, :]
    """ Creating image """
    for _ in range(data_size):
        for entity_type in ['agent','cross','circle','square','none']:
            image = np.tile(background_color, (pixel_size[0], pixel_size[1], 1))
            if entity_type != 'none':
                alpha = masks[entity_type]
                tile = np.tile(entity_colors[entity_type], (pixel_size[0],pixel_size[1], 1))    
                image = alpha*tile + (1-alpha)*image
            
            path = os.path.join(folder,entity_type)
            path = unique_file(path,'.png')
            filename = path[len(folder)+len(os.sep):]

            imageio.imwrite(path, image)
            dataset_csv.write(entity_type+','+filename+'\n')
dataset_csv.close()