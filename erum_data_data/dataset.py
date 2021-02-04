import os
import requests
import numpy as np
from dataclasses import dataclass
from typing import Iterable
from .utils import download_progress, console, _check_md5


@dataclass
class Dataset:
    name: str
    filename: str
    url: str
    md5: str

    datasets_register = set()

    @classmethod
    def __init_subclass__(cls, /, register: bool = True, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if register:
            cls.datasets_register.add(cls)

    @classmethod
    def print_description(cls) -> None:
        print(cls.__doc__)

    @classmethod
    def data_path(cls, path: str, with_filename: bool = True) -> str:
        data_dir = os.path.join(
            os.path.expanduser(path),
            f"{cls.__name__}",
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
        task_id = download_progress.add_task(
            "download",
            filename=cls.filename,
            start=False,
        )
        # This will break if the response headers doesn't contain content length
        download_progress.update(task_id, total=int(response.headers["Content-length"]))
        data_path = cls.data_path(path)
        with open(data_path, "wb") as dest_file:
            download_progress.start_task(task_id)
            for data in response.iter_content(1 << 20):
                dest_file.write(data)
                download_progress.update(task_id, advance=len(data))

    @classmethod
    def load(
        cls,
        split: str = "train",
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
        split: chosse the training or testing set:
            dataset = 'train' or 'test'
        path: directory where the datasets are saved
            path = './datasets'
        force_download: force re-downloading
            force_download = False


        Returns
        -------
        X, y
        X: a list of numpy arrays with X input features - see `print_description()` for more details
        y: a numpy array with labels [0,1]
        """
        if exists := os.path.exists(data_path := cls.data_path(path)):
            if _check_md5(data_path) != cls.md5:
                force_download = True
        if not exists or force_download:
            with download_progress:
                cls.download(path)

        np_zip = np.load(data_path)

        X = []
        for i in range(int(len(np_zip) / 2 - 1)):
            X.append(np_zip["X_{}_{}".format(split, i)])
        y = np_zip["y_{}".format(split)]

        return X, y


def download_datasets(
    datasets: Iterable[str] = Dataset.datasets_register,
    path: str = "./datasets",
    workers: int = 4,
) -> None:
    assert set(datasets) <= Dataset.datasets_register
    with download_progress:
        for dataset in datasets:
            dataset.download(path)