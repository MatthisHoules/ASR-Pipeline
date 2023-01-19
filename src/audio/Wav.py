# External Imports
import os
import filetype
import wave
import numpy as np
import librosa

# Internal Imports
from .utils import time_to_frame



class Wav :
    """
        TODO
    """

    def __init__(self, wav_filepath : str) :
        """
            TODO
        """
        self.filepath : str = wav_filepath
        self.wav_params = self.__retreive_wav_params()

        self.__assert_wav_file()
    # def __init__(self, wav_filepath : str)


    def __retreive_wav_params(self) : 
        """
            TODO
        """

        wavefile = wave.open(self.filepath, 'r')
        return wavefile.getparams()
    # def __retreive_wav_params(self)



    def __assert_wav_file(self) :
        """
            TODO
        """

        assert os.path.exists(self.filepath) and os.path.isfile(self.filepath)
        kind = filetype.guess(self.filepath)
        assert kind is not None and kind.extension == "wav"

        self.filepath : str = self.filepath

        wavefile = wave.open(self.filepath, 'r')
        wav_params = wavefile.getparams()
        wavefile.close()

        assert self.wav_params == wav_params
    # def __assert_wav_file(self)


    
    def get_waveform(self, block_length_s : float = None, stripe_length_s : float = None, mono : bool = True, sr : int = None) -> dict :
        """
            TODO
        """
        
        self.__assert_wav_file()

        wavefile = wave.open(self.filepath, 'r')
        wav_params = wavefile.getparams()
        
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
            "UNKOWN FORMAT ! TODO Throw exception !"

        # Read and convert to float array
        frames = np.frombuffer(wavefile.readframes(wav_params.nframes), dtype = datatype)
        frames = fconverter(np.asarray(frames, dtype = np.float32))

        if wav_params.nchannels > 1 :  
            frames = np.array([frames[offset::wav_params.nchannels] for offset in range(wav_params.nchannels)])
        
        # Multichannel (ex. Stereo) to Mono
        if mono is True :
            frames =  librosa.to_mono(frames)

        # Resample
        if sr is not None : 
            frames = librosa.resample(
                frames, 
                orig_sr = wav_params.framerate, 
                target_sr = sr
            )
        else : 
            sr = wav_params.framerate

        wavefile.close()

        if block_length_s is not None : 
            # No stripe
            if stripe_length_s is None or stripe_length_s == 0 : 
                return self.__get_block_waveform(
                    frames,
                    sr,
                    block_length_s
                )
            # Stripped blocks
            else : 
                return self.__get_striped_block_waveform(
                    frames,
                    sr,
                    block_length_s,
                    stripe_length_s
                )
        
        return frames 
    # def get_waveform(self, block_length_s : float = None, stripe_length_s : float = None, mono : bool = True, sr : int = None) -> dict



    def __get_block_waveform(self, waveform : np.array, sr : int, block_length_s : float) :
        """
            TODO
        """
        assert block_length_s > 0
        assert sr > 0

        nframes : int = waveform.shape[0]
        duration_seconds : float = nframes / sr    
        block_length_frame : int = time_to_frame(block_length_s, duration_seconds, nframes)
        
        start_block_frame : int = 0
        while start_block_frame < nframes :

            end_block_frame  = start_block_frame + block_length_frame

            yield {
                "start_block_frame" : start_block_frame,
                "content" : waveform[start_block_frame:end_block_frame],
                "end_block_frame" : end_block_frame,
                "block_length" : end_block_frame - start_block_frame
            }

            start_block_frame = end_block_frame
    # def __get_block_waveform(self, waveform : np.array, sr : int, block_length_s : float)



    def __get_striped_block_waveform(self, waveform : np.array, sr : int, block_length_s : float, stripe_length_s : float) :
        """
            TODO
        """

        assert block_length_s > 0
        assert stripe_length_s > 0
        assert stripe_length_s <= block_length_s

        nframes : int = waveform.shape[0]
        duration_seconds : float = nframes / sr
            
        stripe_length_frame : int = time_to_frame(stripe_length_s, duration_seconds, nframes)
        block_length_frame : int = time_to_frame(block_length_s, duration_seconds, nframes)

        start_block_frame : int = 0
        while start_block_frame < nframes :

            # Start
            if start_block_frame == 0 :
                start_frame = 0
                is_stripe_left = False
            else :
                if start_block_frame - stripe_length_frame <= 0 :
                    start_frame = 0
                else : 
                    start_frame = start_block_frame - stripe_length_frame
                is_stripe_left = True

            # End
            end : int = start_block_frame + block_length_frame + stripe_length_frame
            end_frame : int = end if end <= nframes else nframes
            is_stripe_right : bool = (end < nframes)
            end_block_frame = start_block_frame + block_length_frame if is_stripe_right else end_frame

            f = waveform[start_frame:end_frame]

            yield {
                "start_frame" : start_frame,
                "is_stripe_left" : is_stripe_left,
                "start_block_frame" : start_block_frame,
                "content" : f,
                "end_block_frame" : end_block_frame,
                "end_frame" : end_frame,
                "is_stripe_right" : is_stripe_right,
                "block_length" : end_block_frame - start_block_frame
            }

            start_block_frame = end_block_frame
    # def __get_striped_block_waveform(self, waveform : np.array, sr : int, block_length_s : float, stripe_length_s : float)

# class Wav