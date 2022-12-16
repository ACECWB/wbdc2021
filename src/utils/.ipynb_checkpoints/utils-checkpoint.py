# import numpy as np
# import torch
my_name_dict = {
#     "ocr": 'ocr_',
#                 "asr": 'asr_',
#                 "des": "des_",
#                 "ocr_char": 'ocr_char_',
#                 "asr_char": 'asr_char_',
                "des_char": "description_char_",
                "tag": "manual_tag_",
                "key": "manual_key_",
                "feed_emb": "feed_embed_",
                }
my_len_dict = {
#     "ocr": 21,
#                "asr": 21,
#                "des": 20,
#                "ocr_char": 41,
#                "asr_char": 41,
#                "des_char": 41,
               "tag": 11,
               "key": 18,
               "feed_emb": 64,
               }
my_vocab_dict = {
#     "ocr": 59775,
#                "asr": 59768,
#                "des": 41241,
#                "ocr_char": 20995,
#                "asr_char": 20870,
#                "des_char": 20988,
               "tag": 350,
               "key": 23262,
               "feed_emb": 64,
                "userid": 199999,
                 "feedid":103785,
                 "authorid":18788,
                 "device":2,
                 "bgm_song_id":5604,
                 "bgm_singer_id":5145
               }
def feature_list(name, mylen=None):
    FEED_LIST = []
    if mylen==None:
        mylen = my_len_dict[name]
    for i in range(mylen):
        FEED_LIST.append(my_name_dict[name] + str(i))
    return FEED_LIST



# # -*- coding:utf-8 -*-
# """

# Author:
#     Weichen Shen,weichenswc@163.com

# """



# def concat_fun(inputs, axis=-1):
#     if len(inputs) == 1:
#         return inputs[0]
#     else:
#         return torch.cat(inputs, dim=axis)


# def slice_arrays(arrays, start=None, stop=None):
#     """Slice an array or list of arrays.

#     This takes an array-like, or a list of
#     array-likes, and outputs:
#         - arrays[start:stop] if `arrays` is an array-like
#         - [x[start:stop] for x in arrays] if `arrays` is a list

#     Can also work on list/array of indices: `slice_arrays(x, indices)`

#     Arguments:
#         arrays: Single array or list of arrays.
#         start: can be an integer index (start index)
#             or a list/array of indices
#         stop: integer (stop index); should be None if
#             `start` was a list.

#     Returns:
#         A slice of the array(s).

#     Raises:
#         ValueError: If the value of start is a list and stop is not None.
#     """

#     if arrays is None:
#         return [None]

#     if isinstance(arrays, np.ndarray):
#         arrays = [arrays]

#     if isinstance(start, list) and stop is not None:
#         raise ValueError('The stop argument has to be None if the value of start '
#                          'is a list.')
#     elif isinstance(arrays, list):
#         if hasattr(start, '__len__'):
#             # hdf5 datasets only support list objects as indices
#             if hasattr(start, 'shape'):
#                 start = start.tolist()
#             return [None if x is None else x[start] for x in arrays]
#         else:
#             if len(arrays) == 1:
#                 return arrays[0][start:stop]
#             return [None if x is None else x[start:stop] for x in arrays]
#     else:
#         if hasattr(start, '__len__'):
#             if hasattr(start, 'shape'):
#                 start = start.tolist()
#             return arrays[start]
#         elif hasattr(start, '__getitem__'):
#             return arrays[start:stop]
#         else:
#             return [None]
