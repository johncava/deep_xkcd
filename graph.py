import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

loss = np.load('loss30.npy')
plt.plot(xrange(1,len(loss) + 1), loss)
plt.xlabel('Iterations')
plt.ylabel('MSE Loss')
plt.title('MSE Loss of Mask-R-CNN-4 [Batch = 3] (6_epoch) lr=5e-4')
plt.show()
plt.savefig('result_loss30.png')
