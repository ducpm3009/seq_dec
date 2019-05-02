import shutil
import os
import re
import cv2

import numpy as np
from six.moves import range

splitter = re.compile("\s+")
base_path = os.getcwd() + '/img/' # Set this path to your folder

def process_folders():
    with open('./data/Eval/list_eval_partition.txt', 'r') as eval_partition_file:
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition]
        # print(list_eval_partition)
        list_all = [(v[0][4:], v[0].split('/')[1].split('_')[-1], v[1]) for v in list_eval_partition]
        # print(list_all)

    # Put each image into the relevant folder in train/test/validation folder
    for element in list_all:
        if not os.path.exists(os.path.join(base_path, element[2])):
            print(os.getcwd())
            os.mkdir(os.path.join(base_path, element[2]))
        if not os.path.exists(os.path.join(os.path.join(base_path, element[2]), element[1])):
            os.mkdir(os.path.join(os.path.join(base_path, element[2]), element[1]))
        if not os.path.exists(os.path.join(os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1])),
                              element[0].split('/')[0])):
            os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1])),
                     element[0].split('/')[0]))
        shutil.move(os.path.join(base_path, element[0]),
                    os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1]), element[0]))

# process_folders()

def create_dict_bboxes(list_all, split='train'):
    lst = [(line[0], line[1], line[3], line[2]) for line in list_all if line[2] == split]
    lst = [("".join(line[0].split('/')[0] + '/' + line[3] + '/' + line[1] + line[0][3:]), line[1], line[2]) for line in lst]
    lst_shape = [cv2.imread(line[0]).shape for line in lst]
    lst = [(line[0], line[1], (round(line[2][0] / shape[0], 2), round(line[2][1] / shape[1], 2), round(line[2][2] / shape[0], 2), round(line[2][3] / shape[1], 2))) for line, shape in zip(lst, lst_shape)]
    dict_ = {"/".join(line[0].split('/')[2:]): {'origin': {'x': line[2][0], 'y': line[2][1]}, 'width': line[2][2], 'height': line[2][3]} for line in lst}
    return dict_

def get_dict_bboxes():
    with open('./data/Anno/list_category_img.txt', 'r') as category_img_file,open('./data/Eval/list_eval_partition.txt', 'r') as eval_partition_file,open('./data/Anno/list_bbox.txt', 'r') as bbox_file:
        list_category_img = [line.rstrip('\n') for line in category_img_file][2:]
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        list_bbox = [line.rstrip('\n') for line in bbox_file][2:]

        list_category_img = [splitter.split(line) for line in list_category_img]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition]
        list_bbox = [splitter.split(line) for line in list_bbox]

        list_all = [(k[0], k[0].split('/')[1].split('_')[-1], v[1], (int(b[1]), int(b[2]), int(b[3]), int(b[4])))
                    for k, v, b in zip(list_category_img, list_eval_partition, list_bbox)]

        list_all.sort(key=lambda x: x[1])

        dict_train = create_dict_bboxes(list_all)
        dict_val = create_dict_bboxes(list_all, split='val')
        dict_test = create_dict_bboxes(list_all, split='test')

        return dict_train, dict_val, dict_test

from keras.models import Model
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K

model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

for layer in model_resnet.layers[:-12]:
    # 6 - 12 - 18 have been tried. 12 is the best.
    layer.trainable = False

x = model_resnet.output
x = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(x) # THIS IS ELU, NOT RELU
y = Dense(46, activation='softmax', name='img')(x)

x_bbox = model_resnet.output
x_bbox = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
x_bbox = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
bbox = Dense(4, kernel_initializer='normal', name='bbox')(x_bbox)

final_model = Model(inputs=model_resnet.input,
                    outputs=[y, bbox])

print(final_model.summary())

opt = SGD(lr=0.0001, momentum=0.9, nesterov=True)

final_model.compile(optimizer=opt,
                    loss={'img': 'categorical_crossentropy',
                          'bbox': 'mean_squared_error'},
                    metrics={'img': ['accuracy', 'top_k_categorical_accuracy'], # default: top-5
                             'bbox': ['mse']})
train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator()

class DirectoryIteratorWithBoundingBoxes(DirectoryIterator):
    def __init__(self, directory, image_data_generator, bounding_boxes: dict = None, target_size=(256, 256),
                 color_mode: str = 'rgb', classes=None, class_mode: str = 'categorical', batch_size: int = 32,
                 shuffle: bool = True, seed=None, data_format=None, save_to_dir=None,
                 save_prefix: str = '', save_format: str = 'jpeg', follow_links: bool = False):
        super().__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size,
                         shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links)
        self.bounding_boxes = bounding_boxes

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        locations = np.zeros((len(batch_x),) + (4,), dtype=K.floatx())

        grayscale = self.color_mode == 'grayscale'
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = image.load_img(os.path.join(self.directory, fname),
                                 grayscale=grayscale,
                                 target_size=self.target_size)
            x = image.img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            if self.bounding_boxes is not None:
                bounding_box = self.bounding_boxes[fname]
                locations[i] = np.asarray(
                    [bounding_box['origin']['x'], bounding_box['origin']['y'], bounding_box['width'],
                     bounding_box['height']],
                    dtype=K.floatx())
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), 46), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x

        if self.bounding_boxes is not None:
            return batch_x, [batch_y, locations]
        else:
            return batch_x, batch_y

dict_train, dict_val, dict_test = get_dict_bboxes()

train_iterator = DirectoryIteratorWithBoundingBoxes("./img/train", train_datagen, bounding_boxes=dict_train, target_size=(200, 200))

test_iterator = DirectoryIteratorWithBoundingBoxes("./img/val", test_datagen, bounding_boxes=dict_val,target_size=(200, 200))

lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               patience=12,
                               factor=0.5,
                               verbose=1)
tensorboard = TensorBoard(log_dir='./logs')
early_stopper = EarlyStopping(monitor='val_loss',
                              patience=30,
                              verbose=1)
checkpoint = ModelCheckpoint('./models/model.h5')

def custom_generator(iterator):
    while True:
        batch_x, batch_y = iterator.next()
        yield (batch_x, batch_y)

final_model.fit_generator(custom_generator(train_iterator),
                          steps_per_epoch=2000,
                          epochs=200, validation_data=custom_generator(test_iterator),
                          validation_steps=200,
                          verbose=2,
                          shuffle=True,
                          callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard],
                          workers=12)


test_datagen = ImageDataGenerator()
test_iterator = DirectoryIteratorWithBoundingBoxes("./img/test", test_datagen, bounding_boxes=dict_test, target_size=(200, 200))

scores = final_model.evaluate_generator(test_iterator.next(), steps=2000)

print('Multi target loss: ' + str(scores[0]))
print('Image loss: ' + str(scores[1]))
print('Bounding boxes loss: ' + str(scores[2]))
print('Image accuracy: ' + str(scores[3]))
print('Top-5 image accuracy: ' + str(scores[4]))
print('Bounding boxes error: ' + str(scores[5]))
