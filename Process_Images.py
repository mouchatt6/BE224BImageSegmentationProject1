import os
import pandas as pd
import numpy as np
from pathlib import Path 
from PIL import Image

def processImages(imgDirectory: str, saveDirectory: str = os.getcwd(), returnDF:bool = False) -> pd.DataFrame | None:
    """
    Process binarized predicted mask images saved in a directory, implement checks, 
    and create a `'submission.csv'` submission file containing image status and mask indices.

    **Do NOT modify this function.**

    Parameters
    ----------
    imgDirectory : str
        The directory containing the images to be processed. It should have exactly 127 .png files.
        When you save your model's predicted masks, make sure the pixel values 
        are either 0 or 255, and save it as a .png file (preferably using PIL).
    saveDirectory : str, optional
        The directory where the resulting DataFrame will be saved as a CSV file. 
        Defaults to the current directory.
    returnDF : bool, optional
        Whether to return the DataFrame. Defaults to False.

    Returns
    -------
    df: A DataFrame with columns 'imageID', 'status', and 'mask', indexed by 'imageID' if `returnDF` is True, else None

    Raises
    ------
    ValueError
        If the number of .png files in `imgDirectory` is not 127,
        if any image is not binary, or if any image is not 512x512 pixels.

    Example Usage
    -------------
    `processImages('path/to/img/folder', 'path/to/save/folder')`
    """

    files = [f for f in os.listdir(imgDirectory) if f.endswith('.png')]  # Get all .png files in the directory
    if len(files) != 127:
        raise ValueError("Directory must contain exactly 127 .png files")

    files.sort(key=lambda x: int(x.split('_')[0]))  # Sort the files

    data = []  # List of dictionaries to be converted to DataFrame
    for file in files:
        imgPath = os.path.join(imgDirectory, file)
        img = np.array(Image.open(imgPath).convert('L'), dtype=np.uint8)

        # Check if image is binary
        if not np.array_equal(img, img.astype(bool).astype(img.dtype) * 255):
            raise ValueError(f"Image {file} is not binary")
        # Check image size
        if img.shape != (512, 512):
            raise ValueError(f"Image {file} is not of size 512x512")

        status = 1 if np.any(img == 255) else 0  # Determine status of image
        maskIndices = ' '.join(map(str, np.nonzero(img.flatten() == 255)[0])) if status else '-100'

        data.append({'imageID': int(file.split('_')[0]), 'status': status, 'mask': maskIndices})

    df = pd.DataFrame(data).set_index('imageID')
    df.to_csv(os.path.join(saveDirectory, 'submission.csv'))

    if returnDF: return df

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    processImages(
        imgDirectory=str(repo_root / "outputs" / "baseline_model" / "hough" / "test_masks"),
        saveDirectory=str(repo_root),
        returnDF=False,
    )
