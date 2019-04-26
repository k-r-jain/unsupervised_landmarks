import torch
import numpy as np
import matplotlib.pyplot as plt

loss_basic = []
loss_transpose = []
for i in range(100):
    try:
        model = torch.load('/home/kartik/Desktop/results_basic_models/basic_celeba_' + str(i) + '.pth')
        loss_basic.append(model['loss'])
        # model = torch.load('/home/kartik/Desktop/results_transpose_models_paper_wts/basic_celeba_' + str(i) + '.pth')
        model = torch.load('/home/kartik/Desktop/results_cars_transpose/transpose_car_' + str(i) + '.pth')
        loss_transpose.append(model['loss'])


    except:
        pass



plt.plot(loss_basic, label = 'Upsample')
plt.plot(loss_transpose, label = 'ConvTranspose')
plt.xlabel('No. of epochs')
plt.ylabel('Perceptual loss (Weighted sum)')
plt.title('Avg. loss (weighted) per epoch')
plt.legend(loc = 'upper right')
plt.show()