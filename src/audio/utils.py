import os
import filetype



def is_file_wav(path : str) -> bool :
    """
        ## is_file_wav

        ### params :
            path : path to check of the file is a wav file or not
        ### return :
            True if the file is a .wav file, False instead
    """

    if not os.path.exists(path) or not os.path.isfile(path) :
        return False
        
    kind = filetype.guess(path)
    return kind is not None and kind.extension == "wav"
# def is_file_wav(path : str) -> bool 