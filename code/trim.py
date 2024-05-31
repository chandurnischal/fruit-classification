import os
import shutil
import random
from tqdm import tqdm

# function to reduce the number of samples in each class
def copyFiles(sourceFolder, destinationFolder):
    if not os.path.exists(destinationFolder):
        os.makedirs(destinationFolder)

    # randomly picks 625 samples from each subdirectory and copies it to destination folder
    files = os.listdir(sourceFolder)
    filesToCopy = random.sample(files, 625)

    for fileName in filesToCopy:
        sourcePath = os.path.join(sourceFolder, fileName)
        destinationPath = os.path.join(destinationFolder, fileName)
        shutil.copyfile(sourcePath, destinationPath)

# path to the folders
sourceFolder = "bigData"
destinationFolder = "data"

if not os.path.exists(destinationFolder):
    os.mkdir(destinationFolder)

classes = os.listdir(sourceFolder)

for c in tqdm(classes):
    sourceClass = os.path.join(sourceFolder, c)
    destinationClass = os.path.join(destinationFolder, c)
    copyFiles(sourceClass, destinationClass)
