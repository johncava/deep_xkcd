import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

loss = np.load('loss35.npy')
plt.plot(xrange(1,len(loss) + 1), loss)
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy Loss')
plt.title('Cross Entropy Loss of CNN-LSTM Model Omega-2 (3_epoch) lr=1e-4')
plt.show()
plt.savefig('result_loss35.png')
