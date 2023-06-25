from keras.callbacks import Callback
class EarlyStoppingMultiple(Callback):
    def __init__(self, monitor1='loss', monitor2='val_loss', patience=0, fav_loss=1.9, fav_val_loss=1.82):
        super(EarlyStoppingMultiple, self).__init__()
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.fav_loss = fav_loss
        self.fav_val_loss = fav_val_loss
        self.patience = patience
        self.wait = 0
        self.best_loss = float('inf')
        self.best_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor1)
        current_val_loss = logs.get(self.monitor2)
        try:
            if current_loss < self.fav_loss and current_val_loss < self.fav_val_loss:
                self.best_loss = current_loss
                self.best_val_loss = current_val_loss
                self.wait = 0
                self.model.stop_training = True
                print(current_loss)
                print(current_val_loss)
            else:
                self.wait += 1
                # if self.wait >= self.patience:
                #     self.model.stop_training = True
        except:
            pass