# External Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def plot_waveform_with_ipus(waveform : np.ndarray, ipus_dataframe : pd.DataFrame) -> None :
    """
        ## plot_waveform_with_ipus

        Plot the wav voice signal by adding the ipus segmentations (ipu start : red vertical bars ; ipu end : blue vertical bars)

        ### inputs :
            waveform : np.ndarray - Voice signal
            ipus_dataframe : pd.DataFrame - IPU segmentation dataframe
        ### Return :
            None
    """

    plt.plot(waveform)
    for _, row in ipus_dataframe.iterrows() :
        plt.axvline(x = row["ipu_start_frame"], color = 'r')
        plt.axvline(x = row["ipu_end_frame"], color = 'b')
    plt.show()
# def plot_waveform_with_ipus(waveform : np.ndarray, ipus_dataframe : pd.DataFrame) -> None