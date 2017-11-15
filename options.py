LENET_OPTION = {'name': 'LeNet',
                'batch_size': 128,
                'epochs': 200,
                'iterations': 391,
                'log_filepath': r'./lenet19/',
                'scheduler': [(60, 0.05), (120, 0.01), (160, 0.002), (200, 0.004)],
                }

VGG_OPTION = {'name': 'vgg19',
              'dropout': 0.5,
              'weight_decay': 0.0005,
              'WEIGHTS_PATH': 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5',
              'batch_size': 128,
              'epochs': 200,
              'iterations': 391,
              'log_filepath': r'./vgg19/',
              'scheduler': [(60, 0.05), (120, 0.01), (160, 0.002), (200, 0.004)],
              }

RESNET_OPTION = {'name': 'resnet',
                 'weight_decay': 0.0005,
                 'batch_size': 128,
                 'epochs': 200,
                 'iterations': 391,
                 'log_filepath': r'./resnet50/',
                 'scheduler': [(60, 0.05), (120, 0.01), (160, 0.002), (200, 0.004)],
                 'stack_n': 18
                 }

DENSENET_OPTION = {'name': 'densenet',
                   'growth_rate': 12,
                   'depth': 100,
                   'compression': 0.5,
                   'weight_decay': 0.0002,
                   'batch_size': 64,
                   'epochs': 250,
                   'iterations': 782,
                   'log_filepath': r'./densenet/',
                   'scheduler': [(75, 0.05), (150, 0.005), (210, 0.0005), (250, 0.0001)],
                   }

RESNEXT_OPTION = {'name': 'resnext',
                  'weight_decay': 0.0005,
                  'cardinality': 4,
                  'base_width': 64,
                  'inplanes': 64,
                  'expansion': 4,
                  'batch_size': 120,
                  'epochs': 250,
                  'iterations': 417,
                  'log_filepath': r'./resnext/',
                  'scheduler': [(75, 0.05), (150, 0.005), (210, 0.0005), (250, 0.0001)],
                  }

XCEPTION_OPTION = {'name': 'xception',
                   'weight_decay': 0.0005,
                   'WEIGHTS_PATH': 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                   'batch_size': 120,
                   'epochs': 250,
                   'iterations': 417,
                   'log_filepath': r'./xception/',
                   'scheduler': [(75, 0.05), (150, 0.005), (210, 0.0005), (250, 0.0001)],
                   }

OPTIONS = {'LeNet': LENET_OPTION, 'Vgg19': VGG_OPTION, 'ResNet': RESNET_OPTION, 'DenseNet': DENSENET_OPTION, 'ResNext': RESNEXT_OPTION,
           'Xception': XCEPTION_OPTION}
