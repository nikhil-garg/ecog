import numpy as np 
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import utils as np_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend import learning_phase
from EEGModels import EEGNet, DeepConvNet
from htnet_model import htnet
import nni

class ReportIntermediates(Callback):
    """
    Callback class for reporting intermediate accuracy metrics.
    This callback sends accuracy to NNI framework every 100 steps,
    so you can view the learning curve on web UI.
    If an assessor is configured in experiment's YAML file,
    it will use these metrics for early stopping.
    """
    def on_epoch_end(self, epoch, logs=None):
        """Reports intermediate accuracy to NNI framework"""
        # TensorFlow 2.0 API reference claims the key is `val_acc`, but in fact it's `val_accuracy`
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_acc'])
        else:
            nni.report_intermediate_result(logs['val_accuracy'])


def main(args):
    path=os.path.join("..","epoched_data/bci3")
    data=np.load(os.path.join(path,"bci3epochs.npz"))
    X=data['X']
    y=data['y']
    X_test=data['X_test']
    y_test=data['y_test']

    kernels, chans, samples = 1, 64, 1000

    y_test       = np_utils.to_categorical((y_test+1)/2)
    y            = np_utils.to_categorical((y+1)/2)

    X            =X.reshape(X.shape[0],kernels,chans, samples)
    X_test       = X_test.reshape(X_test.shape[0], kernels, chans, samples)

    K.set_image_data_format('channels_first')
    f1=args['f1']
    d=args['d']
    f2=f1*d
    model = htnet(useHilbert=True, nb_classes = 2, Chans = chans, Samples = samples, dropoutRate = args['dropout_rate'], kernLength = args['kern_length'], F1 = f1, D = d, F2 = f2, dropoutType = 'Dropout', kernLength_sep=args['kern_length_sep'], norm_rate=args['norm_rate'], data_srate=1000)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=args['lr_decay'],patience=2,verbose=1)
    earlystopper = EarlyStopping(patience=20, verbose=1)
    # compile the model and set the optimizers
    optimizer = tf.keras.optimizers.Adam(learning_rate=args['lr'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics = ['accuracy'])

    checkpointer = ModelCheckpoint(filepath='checkpoint.h5', verbose=1,save_best_only=True)
    model.fit(X, y, batch_size =args['batch_size'], epochs = args['num_epochs'], verbose = 2, validation_data=(X_test, y_test),callbacks=[earlystopper,reduce_lr,checkpointer,ReportIntermediates()])
    model.load_weights('checkpoint.h5')
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    nni.report_final_result(accuracy)


if __name__=='__main__':
    params = {
        'batchs_size': 16,
        'lr': 0.005,
        'lr_decay':0.1,
        'kern_length': 64,
        'kern_length_sep':32,
        'num_epochs': 200,
        'dropout_rate': 0.5,
        'f1' :4,
        'd':2,
        'norm_rate':3
    }
    new_params=nni.get_next_parameter()
    params.update(new_params)

    main(params)

