import numpy as np
import utils
import optimize
import PIL
from PIL import Image

model_name = 'caffenet'
layer = 'fc7'
net = utils.init_model(model_name)

src_fname = 'val_13.JPEG'
guide_fname = 'val_23.JPEG'
src = utils.read_image_rgb(src_fname, net)
guide = utils.read_image_rgb(guide_fname, net)

utils.showarray_noproc(src)

utils.showarray_noproc(guide)

iter_n=10
max_thres=10.
ad, bc, cd = optimize.optimize_single(src_fname, guide_fname, model_name, layer, iter_n, max_thres, net)

diff = ad - src
print np.max(np.abs(diff))

#utils.showarray_noproc(ad)
#utils.showarray_noproc(diff)
#utils.showarray_noproc(src)

img = PIL.Image.fromarray(np.uint8(ad))
#img = Image.fromarray(np.uint8(ad))
#img.save("/home/honggang/Documents/ad5.png")
#img.save("/home/honggang/Documents/"+ '%d' %100 + '_%d' %100 + '.png')

img.save('ad.png')


proc_src = utils.preprocess(net, src)
proc_guide = utils.preprocess(net, guide)

f_src = utils.extract_feat(net, proc_src, layer)
f_guide = utils.extract_feat(net, proc_guide, layer)
print f_src.shape

diff=(f_src - f_guide).flatten()
print "Initial euclidean distance: %.4f" % (np.dot(diff, diff)**.5)

proc_ad = utils.preprocess(net, ad)
f_ad = utils.extract_feat(net, proc_ad, layer)
diff=(f_ad - f_guide).flatten()
print "Final euclidean distance: %.4f" % (np.dot(diff, diff)**.5)