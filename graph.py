import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

loss = np.load('loss11.npy')
plt.plot(xrange(1,len(loss) + 1), loss)
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy Loss')
plt.title('Entropy Loss of CNN-LSTM (2_epoch) lr=1e-3')
plt.show()
plt.savefig('result_loss11.png')
