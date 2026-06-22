import numpy as np


def addchannel_data(data, ch, ndata, section):
    """
    Adds a new channel to the zoo data structure.

    Arguments:
    data -- dict. The zoo data structure.
    ch -- str. The name of the new channel to add.
    ndata -- list or numpy array. The data to add to the new channel.
    section -- str. 'Video' or 'Analog' section.

    Returns:
    data -- dict. The zoo data with the new channel appended.
    """

    # Ensure ch is a string
    if isinstance(ch, list):
        ch = ch[0]

    # Validate the shape of ndata
    if isinstance(ndata, np.ndarray):
        _, c = ndata.shape if ndata.ndim > 1 else (ndata.shape[0], 1)
    else:
        ndata = np.array(ndata)
        _, c = ndata.shape if ndata.ndim > 1 else (len(ndata), 1)

    if c > 3:
        raise ValueError('ndata must be nx1 or nx3')

    if section.lower() == 'video':
        section = 'Video'
    elif section.lower() == 'analog':
        section = 'Analog'

    # Check if the channel already exists
    if ch in data:
        # print(f'WARNING: channel {ch} already exists, overwriting with new data')
        print("WARNING: channel {0} already exists, overwriting with new data".format(ch))

    # Add channel to the zoo structure
    data[ch] = {
        'line': ndata.tolist(),
        'event': {}
    }

    # Add channel to the appropriate channel list
    ochs = data['zoosystem'][section]['Channels']

    if ch not in ochs:
        data['zoosystem'][section]['Channels'] = ochs + [ch]

    return data
