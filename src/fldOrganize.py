import shutil
import os
from src.fileparts import fileparts
from src.engine import engine

def fld_organize(fld, sfld, fn, return_nfld=False):
    subj = fn.split("_")[0]
    sex = fn.split("_")[1]
    cond = fn.split("_")[2].rstrip('.json')

    nfld = os.path.join(fld, sfld, sex,  subj, cond)

    if not os.path.exists(nfld):
        os.makedirs(nfld)

    # move file to new directory
    print(f'moving {fn} to {nfld}')

    if os.path.exists(os.path.join(nfld, fn)):
        print(f"file already exists. Overwriting file: {fn}")
        os.remove(os.path.join(nfld, fn))

    shutil.move(os.path.join(fld, fn), nfld)

    if return_nfld:
        return nfld