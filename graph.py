import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

loss = np.load('loss13.npy')
plt.plot(xrange(1,len(loss) + 1), loss)
plt.xlabel('Iterations')
plt.ylabel('MSE Loss')
plt.title('MSE Loss of Region-CNN w/ ReLU (2_epoch) lr=1e-3')
plt.show()
plt.savefig('result_loss13.png')
