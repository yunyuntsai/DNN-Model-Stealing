import sys
sys.path.insert(1, "/usr/lib/python3/dist-packages")
import caffe
train_file = '/home/yunyuntsai/Documents/Caffe/vgg19_train_val_deepid_mcv_traffic.prototxt'
caffe_model = '/home/yunyuntsai/Documents/Caffe/VGG_ILSVRC_19_layers.caffemodel'

net = caffe.Net(train_file,caffe_model,caffe.TRAIN)

solver_file = 'solver.prototxt'                        #sovler文件保存位置
caffe.set_device(0)                                                      #选择GPU-0
caffe.set_mode_gpu()
solver = caffe.SGDSolver(solver_file)
solver.net.copy_from(caffe_model)
solver.solve()
