import json
from datetime import datetime
import inspect


def zsave(fl, data, message=''):
    """
    Saves zoo files to disk with processing step information appended to the
    zoosystem 'Processing' branch.

    Arguments:
    fl -- str. Full path to file.
    data -- dict. Zoo data.
    message -- str. Further details about the processing step. Default is the current date.
    """
    # Determine which function called zsave
    stack = inspect.stack()
    if len(stack) > 1:
        process = stack[1].function
    else:
        process = 'process'

    # Add additional processing info
    if not message:
        message = ''

    process = f"{process} {message} ({datetime.now().strftime('%Y-%m-%d')})"

    # Write processing step to zoosystem
    if 'Processing' not in data['zoosystem']:
        data['zoosystem']['Processing'] = [process]
    else:
        # Ensure 'Processing' is a list before appending
        if isinstance(data['zoosystem']['Processing'], list):
            data['zoosystem']['Processing'].append(process)
        else:
            # If it is not a list, convert it to a list
            data['zoosystem']['Processing'] = [data['zoosystem']['Processing'], process]

    # Traditional save
    with open(fl, 'w') as f:
        json.dump(data, f, indent=4)
