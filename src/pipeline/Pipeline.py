# External Imports
import pandas as pd
import numpy as np

import time

import os

# Internal Imports
from src.audio import Wav
from src.audio.utils import is_file_wav

from src.transcription import Transcription

from src.pipeline.utils import save_to_sppas_format



class Pipeline(object) :
    """
        # Pipeline

        This class allows the transcription of a wav file in 2 steps:
            - IPUs segmentation
            - For each IPU : transcription
            - save transcription in sppas csv format

        It is possible to transcribe several wav files at once by using the transcript_batch method.
    """

    def __init__(self, transcription_model_name : str) :
        """
            ## __init__

            ### params :
                transcription_model_name : str - Huggingface name of the ASR model (wav2vec2 with LM)
                ex : "bofenghuang/asr-wav2vec2-ctc-french"
        """

        self.__transcription : Transcription = Transcription(transcription_model_name) 
    # def __init__(self, wav_path : str, transcription_model_name : str)



    def transcript_wav_file_with_ipu_segmentation(self, wav_path : str, plot_ipus : bool = False,
        save_sppas : bool = False, save_sppas_path : str = None) -> pd.DataFrame : 
        """
            ## transcript_wav_file_with_ipu_segmentation

            this method allows the IPU segmentation and transcription of a wav file.

            ### params :
                wav_path : str - Wav path 
                plot_ipus : bool - if True : plot the waveform with the IPUs segmentations
                save_sppas : bool - save the transcription in a Sppas readable csv file
                save_sppas_path : str - repository to save the Sppas readable csv file (not used if save_sppas is set to False)

            ### returns :
                pd.DataFrame
        """

        self.set_wav_file(wav_path)

        print("get waveform")
        baseT = time.time()
        waveform : np.ndarray = self.__wav.get_waveform(
            mono = True,
            sr = 16_000
        )
        print("time get waveform : ", time.time() - baseT, " seconds")

        print("get IPUS")
        baseT = time.time()
        df_ipu : pd.DataFrame = self.__wav.get_IPUs(
            waveform, 
            16_000,
            plot=plot_ipus
        )
        print("time get IPUS : ", time.time() - baseT, " seconds")

        list_result : list = list()

        print("Transcription")
        baseT = time.time()
        for _, row in df_ipu.iterrows() :
            # Get ipu waveform
            ipu_waveform : np.ndarray = waveform[int(row.ipu_start_frame) : int(row.ipu_end_frame+1)]

            list_result.append(
                self.__transcription.transcript_ipu(
                    ipu_waveform
                )
            )
        print("time Transcription: ", time.time() - baseT, " seconds")

        df_ipu['transcription'] = list_result

        if save_sppas is True :
            assert save_sppas_path is not None
            save_to_sppas_format(df_ipu, save_sppas_path)

        return df_ipu
    #  def transcript_with_ipu_segmentation(self, plot_ipus : bool = False, word_alignment : bool = True,
    #    save_sppas : bool = False, save_sppas_path : str = None) -> list



    def transcript_batch(self, wav_repository_path : str, plot_ipus : bool = False,
        save_sppas : bool = False, save_sppas_path : str = None) :
        """
            ## transcript_batch

            ### params :
                wav_repository_path : str - repository path of the wav files
                plot_ipus : bool - if True : plot the waveform with the IPUs segmentations for each wav file in wav_repository_path
                save_sppas : bool - save the transcription results for each wav file in wav_repository_path
                save_sppas_path : str - repository to save the Sppas readable csv files (not used if save_sppas is set to False)

        """

        for file in os.listdir(wav_repository_path):
            print("processing file : ", os.path.join(wav_repository_path, file))
            if not is_file_wav(os.path.join(wav_repository_path, file)) :
                continue

            self.transcript_wav_file_with_ipu_segmentation(
                os.path.join(wav_repository_path, file),
                plot_ipus = plot_ipus,
                save_sppas = save_sppas,
                save_sppas_path = os.path.join(save_sppas_path, file + ".csv")
            )
    # def transcript_batch(self, wav_repository_path : str, plot_ipus : bool = False, word_alignment : bool = True,
    #     save_sppas : bool = False, save_sppas_path : str = None) -> list



    def set_wav_file(self, wav_path : str) -> None :
        """
            ## set_wav_file

            __wav private attribute setter

            ### params :
                wav_path : str - path of the wav file
        """

        self.__wav = Wav(wav_path)
    # def set_wav_file(self, wav_path : str) -> None

    
# class Pipeline(class)