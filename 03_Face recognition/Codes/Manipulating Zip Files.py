import zipfile

"""
This is a new tool!
The zipfile library can manilating zip files with some lines of code

"""
zipFilepath  =  "/Users/PedroVitorPereira/Documents/GitHub/Python_" \
                "Projects/course-of-Computer-Vision-Python/03_Face reconition" \
                "/Datasets/yalefaces.zip"

zipObjetc = zipfile.ZipFile(file=zipFilepath, mode= 'r')
zipObjetc.extractall("/Users/PedroVitorPereira/Documents/GitHub/Python"
                     "_Projects/course-of-Computer-Vision-Python/03_Face "
                     "reconition/Datasets")
zipObjetc.close()





