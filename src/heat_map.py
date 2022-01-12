import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_heatmap(y_pred, y_test):
    cf_matrix = confusion_matrix (y_pred, y_test)
    vmin = np.min(cf_matrix)
    vmax = np.max(cf_matrix)
    off_diag_mask = np.eye(*cf_matrix.shape, dtype=bool)

    fig = plt.figure()
    sns.heatmap(cf_matrix, annot=True, mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax)
    sns.heatmap(cf_matrix, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, cbar_kws=dict(ticks=[]))
    plt.show()