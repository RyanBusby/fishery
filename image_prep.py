import numpy as np
from skimage import exposure
from skimage import filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage.transform import resize
from skimage.util import pad

def fix_nv(image):
    '''
    INPUT: numpy.3darray
    OUTPUT: numpy.3darray
    if an image has a green or blue/green tint, changes the correlation of the color channels to reduce the tint
    '''
    h, w, ch = image.shape
    im2 = image.reshape(h*w, 3)
    g_less_r = np.mean(im2, axis=0)[1] - np.mean(im2, axis=0)[0]
    g_less_b = np.mean(im2, axis=0)[1] - np.mean(im2, axis=0)[2]
    if g_less_r > 25 or g_less_b > 25:
        if g_less_b > g_less_r:
            im_adj = image
            im_adj[:,:,2] = image[:,:,2] * 1.6
            im_adj[:,:,1] = np.abs(image[:,:,1].astype(int) - 25).astype(np.uint8)
            im_adj[:,:,1] = image[:,:,1] * .8
        else:
            im_adj = image
            im_adj[:,:,1] = np.abs(image[:,:,1].astype(int) - 40).astype(np.uint8)
            im_adj[:,:,1] = image[:,:,1] * .75
            im_adj[:,:,2] = np.abs(image[:,:,2].astype(int) - 25).astype(np.uint8)
        return im_adj
    else:
        return image

@adapt_rgb(each_channel)
def scharr_each(image):
    '''
    implements skimage scharr filter which finds edges of an image, and adapts the filter to three color channels
    '''
    return filters.scharr(image)

def resize_and_pad(image):
    '''
    INPUT: numpy.3darray
    OUTPUT: numpy.3darray
    reduces the size of an image to 256x144 pixels and keeps the proportions the same by padding images having w/h ratio not equal to 16/9
    '''
    h, w = image.shape[0], image.shape[1]
    if w > h:
        image = resize(image,(144, 144*w/h, 3))
    else:
        image = resize(image,(256*h/w, 256, 3))
    h, w = image.shape[0], image.shape[1]
    h_pad = (256-w)/2
    v_pad = (144-h)/2
    if (256 - w) == 0 and (144 - h) % 2 != 0:
        image = pad(image,((v_pad+1,v_pad),(h_pad,h_pad),(0,0)), 'constant', constant_values=(0,))
    elif(256 - w) % 2 != 0 and (144 - h) == 0:
        image = pad(image,((v_pad,v_pad),(h_pad+1,h_pad),(0,0)), 'constant', constant_values=(0,))
    else:
        image = pad(image,((v_pad,v_pad),(h_pad,h_pad),(0,0)), 'constant', constant_values=(0,))
    return image

def prep_image(image):
    '''
    implement functions and skimage methods to prepare image for processing
    '''
    image = fix_nv(image)
    image = exposure.adjust_gamma(image, gamma=1.2)
    image = exposure.equalize_adapthist(image)
    image = resize_and_pad(image)
    return image
