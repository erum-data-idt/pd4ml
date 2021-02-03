import os
import requests
import numpy as np
from dataclasses import dataclass


from rich.progress import (
    BarColumn,
    DownloadColumn,
    TextColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    Progress,
)


progress = Progress(
    TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    DownloadColumn(),
    "•",
    TransferSpeedColumn(),
    "•",
    TimeRemainingColumn(),
)


@dataclass
class Dataset:
    name: str
    filename: str
    url: str
    md5: str

    @classmethod
    def print_description(cls):
        print(cls.__doc__)

    @classmethod
    def data_path(cls, path: str, with_filename=True) -> str:
        data_dir = os.path.join(
            os.path.expanduser(path),
            f"{cls.__name__}",
            f"{cls.md5}",
        )
        if with_filename:
            return os.path.join(data_dir, f"{cls.filename}")
        else:
            return data_dir

    @classmethod
    def download(cls, path: str) -> None:
        """Copy data from a url to a local file."""

        # handle '~' in path
        datadir = cls.data_path(path, with_filename=False)

        # ensure that directory exists, if not create
        os.makedirs(datadir, exist_ok=True)

        response = requests.get(cls.url, stream=True)
        with progress:
            task_id = progress.add_task(
                "download",
                filename=cls.filename,
                start=False,
            )
            # This will break if the response headers doesn't contain content length
            progress.update(task_id, total=int(response.headers["Content-length"]))
            data_path = cls.data_path(path)
            with open(data_path, "wb") as dest_file:
                progress.start_task(task_id)
                for data in response.iter_content(1 << 20):
                    dest_file.write(data)
                    progress.update(task_id, advance=len(data))

    @classmethod
    def load(
        cls,
        dataset: str = "train",
        path: str = "./datasets",
        force_download: bool = False,
    ):
        f"""
        loads a datafile from a list of options
        Returns a list of X feature numpy arrays for test and training set
        as well as numpy arrays for test and training label

        Additional descriptions of the datasets can be printed via:

        {cls.__name__}.print_description() function

        Parameters
        ----------
        datset: chosse the training or testing set:
            dataset = 'train' or 'test'
        path: directory where the datasets are saved
            path = './datasets'


        Returns
        -------
        X, y
        X: a list of numpy arrays with X input features - see print_decription() for more details
        y: a numpy array with labels [0,1]
        """
        if not os.path.exists(data_path := cls.data_path(path)) or force_download:
            cls.download(path)

        np_zip = np.load(data_path)

        X = []
        for i in range(int(len(np_zip) / 2 - 1)):
            X.append(np_zip["X_{}_{}".format(dataset, i)])
        y = np_zip["y_{}".format(dataset)]

        return X, y
