import pathlib
from abc import ABC, abstractmethod


class VideoProcessingCNN(ABC):

    @abstractmethod
    def prepare_architecture(self, video_data):
        pass

    @abstractmethod
    def train_architecture(self, folder: pathlib.Path):
        pass


class MobileNetCNN(VideoProcessingCNN):

    def prepare_architecture(self, video_data):
        return super().prepare_architecture(video_data)
    
    def train_architecture(self, folder: pathlib.Path):
        return super().train_architecture(folder)
    

class YOLOv8CNN(VideoProcessingCNN):

    def prepare_architecture(self, video_data):
        return super().prepare_architecture(video_data)
    
    def train_architecture(self, folder: pathlib.Path):
        return super().train_architecture(folder)
    

class I3DCNN(VideoProcessingCNN):

    def prepare_architecture(self, video_data):
        return super().prepare_architecture(video_data)
    
    def train_architecture(self, folder: pathlib.Path):
        return super().train_architecture(folder)
    
 
class AudioProcessingCNN(ABC):

    @abstractmethod
    def prepare_architecture(self, audio_data):
        pass

    @abstractmethod
    def train_architecture(self, folder: pathlib.Path):
        pass


class ShuffleNetCNN(AudioProcessingCNN):

    def prepare_architecture(self, audio_data):
        return super().prepare_architecture(audio_data)
    
    def train_architecture(self, folder: pathlib.Path):
        return super().train_architecture(folder)
    

class VGGishCNN(AudioProcessingCNN):

    def prepare_architecture(self, audio_data):
        return super().prepare_architecture(audio_data)
    
    def train_architecture(self, folder: pathlib.Path):
        return super().train_architecture(folder)
    

class EfficientNet_B0CNN(AudioProcessingCNN):

    def prepare_architecture(self, audio_data):
        return super().prepare_architecture(audio_data)
    
    def train_architecture(self, folder: pathlib.Path):
        return super().train_architecture(folder)
    

class CNNFactory(ABC):

    @abstractmethod
    def get_video_cnn(self) -> VideoProcessingCNN:
        pass

    @abstractmethod
    def get_audio_cnn(self) -> AudioProcessingCNN:
        pass


class LightweightCNN(CNNFactory):

    @abstractmethod
    def get_video_cnn(self) -> VideoProcessingCNN:
        return MobileNetCNN()

    @abstractmethod
    def get_audio_cnn(self) -> AudioProcessingCNN:
        return ShuffleNetCNN()


class OptimumCNN(CNNFactory):

    @abstractmethod
    def get_video_cnn(self) -> VideoProcessingCNN:
        return YOLOv8CNN()

    @abstractmethod
    def get_audio_cnn(self) -> AudioProcessingCNN:
        return VGGishCNN()


class ExtensiveCNN(CNNFactory):

    @abstractmethod
    def get_video_exporter(self) -> VideoProcessingCNN:
        return I3DCNN()

    @abstractmethod
    def get_audio_cnn(self) -> AudioProcessingCNN:
        return EfficientNet_B0CNN()


def read_factory() -> CNNFactory:

    factories = {
        "light": LightweightCNN(),
        "optimum": OptimumCNN(),
        "extensive": ExtensiveCNN(),
    }
    while True:
        export_cnn = input("Enter desired architecture CNN advance (light, optimum, extensive): ")
        if export_cnn in factories:
            return factories[export_cnn]
        print(f"Unknown output architecture advance: {export_cnn}.")


def main(fac: CNNFactory) -> None:

    video_cnn = fac.get_video_cnn()
    audio_cnn = fac.get_audio_cnn()

    video_cnn.prepare_architecture("video")
    audio_cnn.prepare_architecture("audio")

    folder = pathlib.Path("/arch_cnn")
    video_cnn.train_architecture(folder)
    audio_cnn.train_architecture(folder)


if __name__ == "__main__":

    factory = read_factory()
    main(factory)

