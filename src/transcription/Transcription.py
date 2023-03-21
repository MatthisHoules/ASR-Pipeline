# External Imports
import torch

from transformers.utils import logging
from transformers import (
    Wav2Vec2ProcessorWithLM,
    Wav2Vec2ForCTC
)

import numpy as np



class Transcription(object) :
    """
        # Transcription
    """



    def __init__(self, model_name : str = "bofenghuang/asr-wav2vec2-ctc-french") :
        """
            ## __init__

            ### params : 
                model_name : str
                    HuggingFace Model ID. Default is : bofenghuang/asr-wav2vec2-ctc-french
                    From https://huggingface.co/bofenghuang/asr-wav2vec2-ctc-french
        """

        print(f"Loading Models : {model_name}")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)
        logging.set_verbosity_error()

        # If Cuda is available
        self.device_available = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device = self.device_available)
    # def __init__(self, model_name : str)



    def transcript_ipu(self, waveform : np.ndarray) -> str :
        """
            ## transcript_ipu

            This method allows to transcript an IPU waveform.

            ### params :
                waveform : np.ndarray - IPU waveform
            ### return :
                str - IPU's transcript
        """

        speech_features = self.processor(
            waveform, 
            sampling_rate = 16_000,
            return_tensors = "pt",
            padding=True
        )   

        with torch.inference_mode():
            logits = self.model(speech_features.input_values).logits

        outputs = self.processor.batch_decode(logits.numpy()).text[0]

        return outputs if len(outputs) > 0 else None
    # def transcript_ipu(self, waveform : np.array, align : bool = True, ipu_offset_s : float = 0)
# class Transcription(object)

    