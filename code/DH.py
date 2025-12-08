import gzip
import numpy as np
import matplotlib as plt

class DataHandler:  
    def __init__(self, Raw_data_folder: str, Processed_data_folder: str):
        import gzip
        import numpy as np
        import matplotlib as plt

        self.raw_data_folder = Raw_data_folder
        self.processed_data_folder = Processed_data_folder

    def getRawDataFolder(self) -> str:
        """Returns current folder that stores raw data"""
        return self.raw_data_folder
    
    def getProcessedDataFolder(self) -> str:
        """Returns current folder that stores processed data"""
        return self.processed_data_folder
    
    def setRawDataFolder(self, new_raw_data_folder: str) -> None:
        """Comment"""
        self.raw_data_folder = new_raw_data_folder

    def setProcessedDataFolder(self, new_processed_data_folder: str) -> None:
        """Comment""" 
        self.processed_data_folder = new_processed_data_folder
    
    def processRawData(self, 
                file: str, 
                file_format: str,
                image_size_x: int = 28,
                image_size_y: int = 28,
                num_images: int = 1,
                ) -> np.ndarray:
        """Preprocesses data for training or validation"""
        #Cases hade kanske varit bättre vid många olika val
        #Effektivt???
        if file_format.upper() == "GZIP":
            return openGzip(file, image_size_x, image_size_y, num_images)
        
        elif file_format.upper() == "CSV":
            pass


    
    def saveProcessedData(self, file: str, name: str) -> None:
        """Stores processed Data"""
        pass



def openGzip(filename: str, image_size_x: int, image_size_y: int, num_images: int) -> np.ndarray:
    """Comment"""
    #Fix if only one size given make it both
    #filename 
    f = gzip.open(filename, "rb")

    #Read pass header
    f.read(16)

    bytes_to_read = image_size_x * image_size_y * num_images
    buf = f.read(bytes_to_read)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, image_size_x, image_size_y)
    return data


#Flytta till passande Class
def showImage(ndArray: np.ndarray, n_image: int, color_scheme: str) -> None:
    """Comment"""
    #Fix error handler for Index error
    try:
        plt.imshow(ndArray[n_image], cmap=color_scheme)
        plt.title(f"Image: {n_image}")
        plt.show()
    except IndexError:
        print("Image not in range")



    
