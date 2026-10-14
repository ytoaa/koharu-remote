"""Prepare SAM-TS-L in ComfyUI's model directory; stdlib only, no pip/CUDA changes."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'dgx'))
from manage_vision import download_files
from vision.text_sam_files import FILES,URLS,REPO,REV,validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir',type=Path,required=True)
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    folder=args.models_dir.expanduser().resolve()/'koharu/text_masks/koharu-text-sam-ts-l'
    if args.download:download_files(folder,REPO,REV,FILES,URLS)
    validate(folder)
    print('SAM_TS_FILES_OK_GPU_NOT_TESTED')
    print(folder)


if __name__=='__main__':main()
