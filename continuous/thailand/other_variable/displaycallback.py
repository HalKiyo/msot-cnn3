import tensorflow as tf
import datetime

class DisplayCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        #self.last_acc = None
        self.last_loss = None
        self.val_loss = None
        self.now_epoch = None

        self.epochs = None

    def print_progress(self):
        epoch = self.now_epoch

        epochs = self.epochs

        print(f"\rEpoch {epoch+1}/{epochs} -- loss: {self.last_loss}  val_loss: {self.val_loss}", end='')

    def on_train_begin(self, logs={}):
        #print('\n#### Train Start #### ' + str(datetime.datetime.now()))

        self.epochs = self.params['epochs']

        self.params['verbose'] = 0

    def on_epoch_begin(self, epoch, log={}):
        self.now_epoch = epoch

    def on_epoch_end(self, epoch, logs={}):
        #self.last_acc = logs.get('acc') if logs.get('acc') else 0.0
        self.last_loss = logs.get('loss') if logs.get('loss') else 0.0
        self.val_loss = logs.get('val_loss') if logs.get('val_loss') else 0.0

        self.print_progress()

    #def on_train_end(self, logs={}):
    #    print('\n#### Train Complete #### ' + str(datetime.datetime.now()))

