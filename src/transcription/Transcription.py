# External Imports
import torch
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)
import numpy as np



class Transcription(object) :
    """
        # Transcription

        Alignment References :
            https://github.com/huggingface/transformers/blob/main/examples/research_projects/wav2vec2/alignment.py
            https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
    """



    def __init__(self, model_name : str) :
        """
            @constructor

            @param : model_name
                HuggingFace Model ID. Default is : bofenghuang/asr-wav2vec2-ctc-french
                From https://huggingface.co/bofenghuang/asr-wav2vec2-ctc-french
        """

        print(f"Loading Models : {model_name}")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

        # If Cuda is available
        self.device_available = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device = self.device_available)
    # def __init__(self, model_name : str)



    def transcript_ipu(self, waveform : np.ndarray) -> list :
        """
            TODO
            TODO list to pd.DataFrame
            ipu_offset_s not use if align = False
        """

        speech_features = self.processor(
            waveform, 
            sampling_rate = 16_000,
            return_tensors = "pt",
            padding=True
        )

        with torch.inference_mode():
            logits = self.model(**speech_features).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        
        outputs = self.processor.batch_decode(predicted_ids)

        return outputs[0] if len(outputs[0]) > 0 else None
    # def transcript_ipu(self, waveform : np.array, align : bool = True, ipu_offset_s : float = 0)
# class Transcription(object)

    