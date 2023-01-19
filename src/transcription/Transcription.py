# External Imports
import sys
import torch
import librosa
import librosa.display
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)
import time
import matplotlib.pyplot as plt
import soundfile as sf
import pyfoal
import numpy as np
from librosa import resample



class Transcription :
    """
        # Transcription

        Alignment References :
            https://github.com/huggingface/transformers/blob/main/examples/research_projects/wav2vec2/alignment.py
            https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
    """



    def __init__(self, model_id) :
        """
            @constructor

            @param : model_name
                HuggingFace Model ID. Default is : bofenghuang/asr-wav2vec2-ctc-french
                From https://huggingface.co/bofenghuang/asr-wav2vec2-ctc-french
        """

        print(f"Loading Models : {model_id}")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)

        print(self.processor.tokenizer.get_vocab().keys())

        # If Cuda is available
        self.device_available = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device = self.device_available)
    # def __init__(self)



    def process_striped_block_waveform(self, inputs : list, sr : int, strip_duration : int) :
        """
            TODO
        """

        chuncks_logits = self.__process_chunks(
            inputs,
            sr,
            strip_duration
        )

        baseT = time.time()
        print("Decoding...")

        logits = torch.cat(list(chuncks_logits), dim=1)

        print("gathered logits tenser size : ", logits.size())
        
        predicted_ids = torch.argmax(logits, dim=-1)
        outputs = self.processor.batch_decode(predicted_ids, output_word_offsets=True)
        transcription : str = outputs.text

        print(outputs)


        # outputs = self.processor.decode(logits[0].cpu().numpy(), output_word_offsets=True)
        # transcription : str = outputs.text

        # Word Aligment with Wav2vec2 (0.02 precision)
        time_offset : float = self.model.config.inputs_to_logits_ratio / self.processor.feature_extractor.sampling_rate # documentation data
        words : list = [
            {
                "word": d["word"],
                "start_time": round(d["start_offset"] * time_offset, 5),
                "end_time": round(d["end_offset"] * time_offset, 5),
                "start_logit" : d["start_offset"], 
                "end_logit" : d["end_offset"]
            }
            for d in outputs.word_offsets[0]
        ]

        # self.__plot_aligment(resampled_waveform, words)
        print("Decoding Done ! Duration : ", time.time() - baseT, " s")

        return transcription, words 
    # def process_striped_block_waveform(self, inputs : list, sr : int, strip_duration : int)



    def __process_chunks(self, inputs, framerate : int, strip_duration_s : float) : 
        """
            TODO
        """
        
        i = 1
        for chunk in inputs : 
            baseT = time.time()

            if framerate != 16_000 : 
                waveform : np.array = resample(
                    chunk["content"],
                    framerate,
                    16_000,
                    fix=True
                )
                
            else : 
                waveform = chunk["content"]

            speech_features = self.processor(
                waveform, 
                sampling_rate = 16_000,
                return_tensors = "pt",
                padding=True
            )

            with torch.no_grad():
                logits = self.model(**speech_features).logits

            # Remove stripped parts
            model_time_offset : float = self.model.config.inputs_to_logits_ratio / self.processor.feature_extractor.sampling_rate

            # Remove Left Strip
            if chunk["is_stripe_left"] is True or chunk["is_stripe_right"] is True :
                print("Strip duration logit : ", (strip_duration_s / model_time_offset))
                strip_duration_logit : int = int(strip_duration_s / model_time_offset)

                print("nb strip logits : ", strip_duration_logit)

                print(type(strip_duration_logit))
            
                if chunk["is_stripe_left"] is True :
                    logits = logits[:,strip_duration_logit:] 
                # Remove Right Strip
                if chunk["is_stripe_right"] is True : 
                    logits = logits[:,:-strip_duration_logit]

            print(f"Compute chunk {i} Done ! Duration : ", time.time() - baseT, " s")

            i += 1

            yield logits
    # def __process_chunk(self, inputs, sr, strip_duration_s)

 

    def __plot_aligment(self, speech_array, speech_text) :
        """
            TODO
        """

        plt.plot(data= librosa.display.waveshow(speech_array, sr=16_000, max_points=1000000))
        
        for word in speech_text:
            x0 = word["start_time"]
            x1 = word["end_time"]
            plt.axvspan(x0, x1, alpha=0.1, color="red")

            w = word["word"]
            plt.annotate(w, (x0, 0.8))

        plt.show()



    def display_words(self, wav_path, words):
        """
            @private_method : __display_words

            CAUTION : this method has been created only for the Pipeline Evaluation.
                Do not use it in production (time and Disk Storage Consuming)
        """

        i = 0
        for word in words :
            x0 = word["start_time"]
            x1 = word["end_time"]
            word = word["word"]

            s, sr = librosa.load(wav_path, offset=x0, duration=x1-x0)

            sf.write(f"./tests/tmp/file_{i}_{word}.wav", s, sr, subtype='PCM_24')

            i += 1
# class Transcription




                # outputs = self.processor.decode(logits[0].cpu().numpy(), output_word_offsets=True)
                # transcription : str = outputs.text


                # # Word Aligment with Wav2vec2 (0.02 precision)
                # time_offset : float = self.model.config.inputs_to_logits_ratio / self.processor.feature_extractor.sampling_rate # documentation data
                # words : list = [
                #     {
                #         "word": d["word"],
                #         "start_time": round(d["start_offset"] * time_offset, 5),
                #         "end_time": round(d["end_offset"] * time_offset, 5),
                #         "start_logit" : d["start_offset"], 
                #         "end_logit" : d["end_offset"]
                #     }
                #     for d in outputs.word_offsets
                # ]

                # self.__plot_aligment(resampled_waveform, words)

                # return transcription, words 