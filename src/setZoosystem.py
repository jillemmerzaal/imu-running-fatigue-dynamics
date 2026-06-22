def setZoosystem(fl):
    """
    Creates 'zoosystem' branch for data being imported to biomechZoo

    Arguments:
    fl -- str. The file name to set in the zoosystem

    Returns:
    zoosystem -- dict. The zoosystem dictionary with appropriate parameters loaded
    """

    # set defaults
    ver = False
    zch = ['Analog', 'Anthro', 'AVR', 'CompInfo', 'SourceFile', 'Units', 'Version', 'Video']
    # Set up structure
    zoosystem = {key: {} for key in zch}

    sections = ['Video', 'Analog']

    for section in sections:
        zoosystem[section] = {
            'Channels': [],
            'Freq': None,
            'Indx': None,
            'ORIGINAL_START_FRAME': None,
            'ORIGINAL_END_FRAME': None,
            'CURRENT_START_FRAME': None,
            'CURRENT_END_FRAME': None
        }

    zoosystem['Processing'] = ''
    zoosystem['AVR'] = 0
    zoosystem['Analog']['FPlates'] = {
        'CORNERS': [],
        'NUMUSED': 0,
        'LOCALORIGIN': None,
        'LABELS': []
    }

    zoosystem['Version'] = ver
    zoosystem['SourceFile'] = str(fl)

    zoosystem['Units'] = {
        'Markers': 'mm',
        'Angles': 'deg',
        'Forces': 'N',
        'Moments': 'Nmm',
        'Power': 'W/kg',
        'Scalars': 'mm'
    }

    return zoosystem
