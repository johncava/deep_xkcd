import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

loss = np.load('loss10.npy')
plt.plot(xrange(1,len(loss) + 1), loss)
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy Loss')
plt.title('Entropy Loss of CNN-LSTM Attention with more Linear Layers (4_epoch) lr=1e-4')
plt.show()
plt.savefig('result_loss10.png')
