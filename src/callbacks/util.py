from pytorch_lightning.callbacks import Callback
import subprocess


class GifCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_end(self, *args, **kargs) -> None:
        subprocess.call(
            [
                "ffmpeg",
                "-f",
                "image2",
                "-framerate",
                "4",
                "-i",
                "results/%d.jpg",
                "video.gif",
            ]
        )
