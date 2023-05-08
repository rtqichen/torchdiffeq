import pickle
import pandas as pd
from matplotlib import pyplot as plt
if __name__ == '__main__':

    em_list = pickle.load(open(f"bp_loss_time_tracker_FVDP.pkl", "rb"))

    df = pd.DataFrame.from_records(data=em_list, columns=['time', 'loss'])
    x = df['time']
    y = df['loss']
    #

    plt.plot(x,y)

    plt.savefig(f'loss_curve.png')
