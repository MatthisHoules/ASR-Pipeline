# External Imports
import pandas as pd
import csv



def save_to_sppas_format(df : pd.DataFrame, save_path : str) -> None :
    """
        ## save_to_sppas_format

        This function allows to save a transcription dataframe to a SPPAS (.csv) readable format.

        ### params :
            df : pd.DataFrame : transcription dataframe to save in SPPAS format.
            save_path : str : file path to save the csv file.
        ### return :
            None  
    """
    df = df.dropna()
    
    df = df[["ipu_start_s", "ipu_end_s", "transcription"]]

    df_silence = pd.DataFrame()
    df_silence["ipu_end_s"] = df["ipu_start_s"]
    df_silence["ipu_start_s"] = df["ipu_end_s"].shift(1)
    df_silence.at[0, "ipu_start_s"] = 0
    df_silence.at[0, "ipu_end_s"] = df.head(1).ipu_start_s
    df_silence["transcription"] = '#'

    results = pd.concat([df, df_silence], ignore_index=True).sort_values("ipu_start_s").reset_index(drop=True)

    # Case speech starts at 0.0s
    if results.at[1, "ipu_end_s"] == 0 and results.at[1, "ipu_start_s"] == 0 :
        results = results.drop(index=results.index[1], axis=0).reset_index(drop=True)

    results["init"] = "IPUs"

    results = results.dropna()
   
    results.to_csv(
        save_path, 
        columns=["init", "ipu_start_s", "ipu_end_s", "transcription"], 
        header=None,
        index=None, 
        quoting=csv.QUOTE_NONNUMERIC
    )
# def save_to_sppas_format(df : pd.DataFrame, save_path : str) -> None