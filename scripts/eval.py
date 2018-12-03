
from pylab import *
import pandas as pd
from scipy import interpolate

train = pd.read_csv('results/train_metrics.csv')
test = pd.read_csv('results/test_metrics.csv')

plot_batch_loss = train['loss'].values
plot_train_acc = train['acc'].values
plot_test_acc = test['acc'].values

ltrain = len(plot_train_acc)
ltest = len(plot_test_acc)
x = np.arange(0, ltest)
f = interpolate.interp1d(x, plot_test_acc)
xnew = np.arange(0, ltest-0.8, 0.2)
plot_test_acc_new = f(xnew)

figure(1, figsize=(16, 9))
subplot(211)
title('Pretraining Accuracy')
plot(plot_train_acc, label='train')
plot(plot_test_acc_new, label='test')
ylim(0.1, 1.1)
grid()
legend()
subplot(212)
title('Loss')
plot(plot_batch_loss, label='loss')
ylim(0.25, 1.1)
grid()
legend()

train_text_no_pretraining = pd.read_csv('results/train_metrics_termtext_nopretrain.csv')
test_text_no_pretraining = pd.read_csv('results/test_metrics_termtext_nopretrain.csv')
train_text_with_pretraining = pd.read_csv('results/train_metrics_termtext_pretrainednetwork.csv')
test_text_with_pretraining = pd.read_csv('results/test_metrics_termtext_pretrainednetwork.csv')

train_loss_no = train_text_no_pretraining['loss'].values
train_loss_with = train_text_with_pretraining['loss'].values

train_acc_no = train_text_no_pretraining['acc'].values
train_acc_with = train_text_with_pretraining['acc'].values
test_acc_no = test_text_no_pretraining['acc'].values
test_acc_with = test_text_with_pretraining['acc'].values


ltrain = len(train_acc_no)
ltest = len(test_acc_no)
print(ltrain)
print(ltest)
print(ltrain / float(ltest))

x = np.arange(0, ltest)
f = interpolate.interp1d(x, test_acc_no)
f2 = interpolate.interp1d(x, test_acc_with)
xnew = np.arange(0, ltest-0.8, 0.5)

test_acc_no_new = f(xnew)
test_acc_with_new = f2(xnew)

figure(2, figsize=(16, 9))
subplot(211)
title('Termtext Accuracy')
plot(train_acc_no, label='train with no pretraining')
plot(train_acc_with, label='train pretrained')
plot(test_acc_no_new, label='test with no pretraining')
plot(test_acc_with_new, label='test pretrained')
ylim(0.1, 1.1)
grid()
legend()
subplot(212)
title('Loss')
plot(train_loss_no, label='no pretraining')
plot(train_loss_with, label='pretrained')
ylim(0.25, 1.1)
legend()
grid()
show()