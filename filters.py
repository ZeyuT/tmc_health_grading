import glob
import numpy as np
import os
import collections

from scipy import signal
from scipy.ndimage import uniform_filter1d




def sliding_filter(arr, window_size, std_t, median_t):
    """Applies a sliding window filter based on median values.
       Windows with average higher than threshold are passed      
    Args:
        arr (numpy.ndarray): The input array.
        window_size (int): The size of the sliding window.
        threshold: for marking start and end points on average array

    Returns:
        list: list of [start_idx, end_idx] for segments passing the filter.
    """

    view = np.lib.stride_tricks.sliding_window_view(arr, window_size)
    median_values = np.median(view, axis=1)
    std_values = np.std(view, axis=1)
    mask = (median_values>median_t) & (std_values>std_t)
    indices = np.where(np.diff(mask>0))[0] + window_size//2
    return median_values, std_values, [[start, end] for start, end in zip(indices[::2], indices[1::2])]

def hysteresis_threshold_segment(signal, start_threshold, end_threshold, interval=5, min_segment=100.):
    """ Find valid segments with high magnitude from a 1D signal using hysteresis thresholds    
    Args:
        signal: the input array.
        start_threshold: high threshold of the beginning of segmentation
        end_threshold: low threshold of the end of segmentation
        interval: number of datapoints skipped to the next evaluated datapoints
        min_segment: the minimum length of detected segments in unit of datapoints.

    Returns:
        list: list of [start_idx, end_idx] for segments passing the filter.
    """
    result_ls = []

    segment_starts = []
    segment_ends = []
    
    state, start, end = 0, 0, 0
    pause_counter = 0
    t_pause = 100

    for i in range(0, len(signal), interval):
        value = signal[i]

        if state == 0 and value > start_threshold:
            state = 1
            start = i
        elif state == 1 and value < end_threshold:
            state = 2
            end = i+1 # for Python list slicing
            pause_counter = 0
        elif state == 2:
            if value > start_threshold:
                state = 1
            else:
                pause_counter += interval
                if pause_counter >= t_pause:
                    if end-start >= min_segment:
                        # save data
                        segment_starts.append(start)
                        segment_ends.append(end)
                    end = 0
                    state = 0
    if state == 1: # catch segment if it ends at the end of probabilities
        end = i
        if pause_counter >= t_pause and end-start >= min_segment:
            # save data
            segment_starts.append(start)
            segment_ends.append(end)
        
                            
    return [[s,e] for s, e in zip(segment_starts, segment_ends)]

def rolling_slope(data, window_size):
    """Calculates the rolling slope of a 1D array using NumPy."""
    view = np.lib.stride_tricks.sliding_window_view(data, window_size)
    x = np.arange(window_size)
    slopes = np.apply_along_axis(lambda y: np.polyfit(x, y, 1)[0], 1, view)
    return slopes


def sliding_slope_filter(arr, window_size, start_t, end_t, min_segment=200):
    """Applies a sliding window filter to calculate rolling slopes. 
        Then output segments of the input array, each contain a dynamic cycle.     
    Args:
        arr (numpy.ndarray): The input array.
        window_size (int): The size of the sliding window.
        start_t: min slope threshold for marking start points of segments 
        end_t: max slope threshold for marking end points of segments 
        min_segment: min length of valid segments
    Returns:
        list: list of [start_idx, end_idx] for segments passing the filter.
    """
    smoothed = uniform_filter1d(arr, size=200) # size 200 = 1 sec
    slopes = rolling_slope(smoothed, window_size)
    start_idxs = np.where(np.diff(slopes-start_t>0))[0][::2] + window_size//2
    end_idxs = np.where(np.diff(slopes-end_t<0))[0][1::2] + window_size//2
    return slopes, [[start, end] for start, end in zip(start_idxs, end_idxs) if end-start>min_segment]