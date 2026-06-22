import os

from src.engine import engine
from src.fileparts import fileparts
from src.setZoosystem import setZoosystem
from src.grab import grab
from src.add_channel import addchannel_data
from src.zsave import zsave
from src.fldOrganize import fld_organize



## hardcode some variables
CONS = ["Pre", "Post"]
SENSORS = ["lumbar"]#, "LLL", "LUL", "RLL", "RUL", "LF", "RF"]
DIR = ['X', 'Y', 'Z']
SAMPLE_RATE = 128

def extract(fld):
    fl = engine(path=fld, extension=".mat")

    for f in fl:
        directory, filename, extension = fileparts(f)
        ## extract subject name and sex from the filename
        subj = filename.split("_")[0]
        sex = filename.split("_")[1]

        for c in CONS:
            fname = f"{subj}_{sex}_{c}.json"
            print(f"Create new json file: {fname}")
            data = {}
            data['zoosystem'] = setZoosystem(fname)
            data['zoosystem']['Units'] = {}
            data['zoosystem']['Video']["Freq"] = SAMPLE_RATE
            data['zoosystem']['AVR'] = 0

            #load data
            all_data = grab(f)

            chns = [f"Acc{c}_{s}" for s in SENSORS]
            try:
                for ch in chns:
                    if "data40" in directory:
                        temp_data = all_data["kinedata"]["Forty"][ch].T
                    else:
                        temp_data = all_data["kinedata"][ch].T

                    nch = ch.replace(c,"")
                    for i, d in enumerate(DIR):
                        chname = f"{nch}_{d}"
                        print(f"    Adding channel: {chname}")
                        ndata = temp_data[i]
                        data = addchannel_data(data, f'{chname}', ndata, 'Video')

                zsave(os.path.join(fld, fname), data)

                fld_organize(fld, "mat2zoo", fname)
            except KeyError as e:
                print(e)