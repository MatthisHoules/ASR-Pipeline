# Exteral Imports
import os
import filetype

import numpy as np
import pandas as pd

import librosa
import wave

# Internal Imports
from . import plot_waveform_with_ipus



class Wav(object) :
    """
        # Wav

        This class allows to load .wav file and extract its waveform.
        The waveform extraction method waveform (Wav.get_waveform) have a lot of parameter for waveform transformation. Please see its documentation
        A method (Wav.get_IPUs) segments the waveform in differents IPUs (Inter Pausal Units)
    """



    def __init__(self, filepath : str) :
        """
            ## __init__

            Wav class constructor

            ### Params : 
                filepath : str - path of the .wav file to load.
        """

        self.filepath : str = filepath
        """
            ### filepath
                str

            Path of the .wav file
        """

        assert os.path.exists(self.filepath) and os.path.isfile(self.filepath)
        kind = filetype.guess(self.filepath)
        assert kind is not None and kind.extension == "wav"

        # Retrieve wav_params
        with wave.open(self.filepath, 'r') as wavefile :
            self.wav_params = wavefile.getparams()
            """
                ### wav_params
                    _wave_params

                Params of the .wav file (!! use this with caution with processed waveform !!)
                extracted with the wave python package.
            """
            print(wavefile.getparams())

        self.duration_s : float = self.wav_params.nframes / self.wav_params.framerate
        """
            ### duration_s
                float

            Duration of the .wav file in seconds.
        """
    # def __init__(self, filepath : str)


        
    def get_waveform(self, mono : bool = True, sr : int = None) -> np.ndarray :
        """
            ## get_waveform

            This method allows to extract the waveform of the .wav file.
                It is also possible to convert the waveform into a mono channel format 
                and change the framerate of the waveform

            ### params : 
                mono : bool (default value : True) - Convert the .wav
                sr : int (nullable, default value : None) - new framerate of the waveform
            ### returns :
                np.ndarray - load and processed waveform
        """

        wavefile = wave.open(self.filepath, 'r')
        
        fconverter = lambda a : a
        if wavefile.getsampwidth() == 1:
            # 8-Bit format is unsigned.
            datatype = np.uint8
            fconverter = lambda a : ((a / 255.0) - 0.5) * 2
        elif wavefile.getsampwidth() == 2:
            # 16-Bit format is signed.
            datatype = np.int16
            fconverter = lambda a : a / 32767.0
        else : 
            raise Exception("Unknown Wav file format.")

        # Read and convert to float array
        frames = np.frombuffer(wavefile.readframes(self.wav_params.nframes), dtype = datatype)
        frames = fconverter(np.asarray(frames, dtype = np.float32))

        if self.wav_params.nchannels > 1 :  
            frames = np.array([frames[offset::self.wav_params.nchannels] for offset in range(self.wav_params.nchannels)])

        # Multichannel (ex. Stereo) to Mono
        if mono is True :
            frames =  librosa.to_mono(frames)

        # Resample
        if sr is not None : 
            frames = librosa.resample(
                frames, 
                orig_sr = self.wav_params.framerate, 
                target_sr = sr
            )
        else : 
            sr = self.wav_params.framerate
        
        wavefile.close()
        return frames
    # def get_waveform(self, mono : bool = True, sr : int = None) -> np.ndarray



    def get_IPUs(self, waveform : np.ndarray, waveform_sr : int, silence_threshold : float = 0.008, ipu_threshold_s : float = .2, plot : bool = False) -> pd.DataFrame :
        """
            ## get_IPUs

            ### params : 
                waveform : np.ndarray - Waveform
                waveform_sr : int - framerate of the waveform passed in params
                silence_threshold : float - silence threshold (if energy > threshold : speech else silence)
                ipu_threshold_s : float - minimum duration in seconds between 2 IPUs
                plot : bool (optionnal, default value : False) - plot the waveform with the IPUs segments
            ### return
                pd.DataFrame - pandas DataFrame of the IPUs segments
                columns : 
                    "ipu_start_s" : start timestamp (in seconds) of the ipu
                    "ipu_start_frame" : start frame of the ipu
                    "ipu_end_s" : end timestamp (in seconds) of the ipu
                    "ipu_end_frame" : start frame of the ipu
        """

        # Root Mean Square Energy
        rms = librosa.feature.rms(y=waveform, frame_length=512)[0]
        df : pd.DataFrame = pd.DataFrame(rms, columns=["rms"])
        
        # Times of each RMSE
        times = np.append(librosa.times_like(rms, sr=waveform_sr), self.duration_s)
        df["start_s"] = times[0:-1]
        df["end_s"] = times[1:]
        df["is_silence"] = df["rms"].lt(silence_threshold)

        # Previous row (lag 1)
        df[["prev_rms", "prev_is_silence"]] = df[["rms", "is_silence"]].shift(1).fillna(False)

        # State transition dataframe (speech to silence) or (silence to speech)
        state_transition_df : pd.DataFrame = df.loc[
            ((df["prev_is_silence"] == False) & (df["is_silence"] == True)) | \
            ((df['prev_is_silence'] == True) & (df['is_silence'] == False))
        ].reset_index(drop=True)

        list_silence_duration : list = []
        for i in range(0,len(state_transition_df)-1) :
            row = state_transition_df.iloc[i]
            n_row = state_transition_df.iloc[i + 1]

            if row["is_silence"] == False : continue
            
            # Silence 
            duration_silence : float = n_row["start_s"] - row["start_s"]
            if duration_silence >= ipu_threshold_s or i == 0 :
                list_silence_duration.append({
                    "pause_start_s" : row["start_s"],
                    "pause_end_s" : n_row["start_s"],
                    "pause_duration_s" : duration_silence
                })

        df_silence : pd.DataFrame = pd.DataFrame(list_silence_duration)
        df_silence["pause_start_frame"] = librosa.time_to_samples(times=df_silence["pause_start_s"], sr=16_000)
        df_silence["pause_end_frame"] = librosa.time_to_samples(times=df_silence["pause_end_s"], sr=16_000)

        # Silence dataframe to IPU dataframe
        df_ipu : pd.DataFrame = pd.DataFrame()
        df_ipu["ipu_start_s"] = df_silence["pause_end_s"]
        df_ipu["ipu_start_frame"] = df_silence["pause_end_frame"]
        df_ipu["ipu_end_s"] = df_silence["pause_start_s"].shift(-1)
        df_ipu["ipu_end_frame"] = df_silence["pause_start_frame"].shift(-1)
        df_ipu.at[len(df_ipu) - 1, "ipu_end_frame"] = len(waveform)
        df_ipu.at[len(df_ipu) - 1, "ipu_end_s"] = len(waveform) / 16000

        if plot is True : 
            plot_waveform_with_ipus(waveform, df_ipu)
        
        return df_ipu
    # def get_IPUs(self, silence_threshold : float = 0.008, ipu_threshold_s : float = .2, plot : bool = False) -> pd.DataFrame 
# class Wav(object)