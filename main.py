# External Imports
import pandas as pd
import sys

# Internal Imports
from src.pipeline import Pipeline
from src.transcription import Transcription



def __main__() -> None : 
    """
        ## __main__

        Main function

        ### return :
            None
    """
    input_wav_files_path : str = sys.argv[1]
    output_csv_path : str = sys.argv[2]

    pipeline : Pipeline = Pipeline(
        transcription_model_name="bofenghuang/asr-wav2vec2-ctc-french"
    )

    pipeline.transcript_batch(
        input_wav_files_path,
        save_sppas=True,
        save_sppas_path = output_csv_path
    )
# def __main__() -> None 



if __name__ == "__main__" :
    __main__()