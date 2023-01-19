

def time_to_frame(time_s : int, total_duration : float, n_frames : int) -> int :
    """
        TODO
    """

    return int((time_s * n_frames) / total_duration)
# def time_to_frame(time_s : int, frame_rate : int, nframes : int) -> int