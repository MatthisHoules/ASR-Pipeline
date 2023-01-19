# External Imports
import sys
import numpy as np
import pandas as pd
import time

# Internal Imports
from src.transcription.Transcription import Transcription
from src.audio.Wav import Wav


FILENAME : str = "./tests/AC.wav"

# FILENAME : str = "./tests/sardoche_angry.wav"

def __main__() -> None : 
    """
        Main Function
    """

    transcription = Transcription("bofenghuang/asr-wav2vec2-ctc-french")

    wav_file : Wav = Wav(FILENAME)

    strip_duration_s : float = 5
    block_duration_s : float = 60

    waveform : np.array = wav_file.get_waveform(
        block_length_s=block_duration_s,
        stripe_length_s=strip_duration_s,
        mono=True,
        sr=16_000
    )

    a, e = transcription.process_striped_block_waveform(
        waveform,
        16_000,
        strip_duration_s
    )

    print(a)
    print()
    print(e)

    # transcription.display_words(FILENAME, e)

    df : pd.DataFrame = pd.DataFrame(e)
    df.to_csv("./tests/ac_part_no_lm.csv", index=None)
    # for c in t :
    #     print("Memory size : ", sys.getsizeof(c))


    # print(t)
    # a = 0
    # for c in t :
    #     a += c["block_length"]
    # #     print()

    # print("Block length : ", a)
# # def __main__()


    
if __name__ == "__main__" :
    baseT = time.time()
    __main__()
    print("all process duration : ", time.time() - baseT, "seconds")