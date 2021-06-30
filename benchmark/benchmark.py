import os
import json
from abc import abstractmethod, ABCMeta
from datetime import datetime
from pd4ml.utils import console
from .infos import meta, runtime, env


class InfoJson(metaclass=ABCMeta):
    def __init__(self, cwd):
        self.cwd = cwd

    @property
    def full_path(self):
        return os.path.join(self.cwd, self.filename)

    @abstractmethod
    def content(self):
        pass

    def write(self):
        with open(self.full_path, self.mode) as f:
            json.dump(self.content(), f, indent=4)


MetaInfo = type(
    "MetaInfo",
    (InfoJson,),
    dict(
        filename="meta.json",
        mode="w",
        content=meta,
    ),
)
RuntimeInfo = type(
    "RuntimeInfo",
    (InfoJson,),
    dict(
        filename="runtime.json",
        mode="w",
        content=runtime,
    ),
)
EnvInfo = type(
    "EnvInfo",
    (InfoJson,),
    dict(
        filename="env.json",
        mode="w",
        content=env,
    ),
)


class Benchmark:
    datefmt = "%Y-%m-%d_%H-%M-%S"

    def __init__(self, dataset, network):
        self.dataset = dataset
        self.network = network
        self.datetime = datetime.now().strftime(self.datefmt)
        console.log(f"Starting new benchmark @ [cyan]{self.cwd}")

    @property
    def wd(self):
        return f"benchmarks/{self.dataset}/{self.network}"

    @property
    def cwd(self):
        return os.path.join(self.wd, f"{self.datetime}")

    def snapshot(self, history={}):
        os.makedirs(self.cwd, exist_ok=True)
        for info in MetaInfo, RuntimeInfo, EnvInfo:
            info = info(cwd=self.cwd)
            console.log(f"Snapshotting [cyan]{info.full_path}...")
            info.write()
        assert isinstance(history, dict)
        HistoryInfo = type(
            "HistoryInfo",
            (InfoJson,),
            dict(
                filename="history.json",
                mode="w",
                content=lambda self: {k: [v[-1]] for k, v in history.items()},
            ),
        )
        info = HistoryInfo(cwd=self.cwd)
        console.log(f"Snapshotting [cyan]{info.full_path}...")
        info.write()

    @property
    def records(self):
        return sorted(
            os.listdir(self.wd),
            key=lambda x: datetime.strptime(x, self.datefmt),
            reverse=True,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_')])}) @ {self.wd}"

    def summary_report(self, sort_by="Timestamp"):
        import pandas as pd
        from rich.table import Table

        data = pd.DataFrame()
        for rec in self.records:
            history = pd.read_json(os.path.join(self.wd, rec, "history.json"))
            history["Timestamp"] = rec
            data = data.append(history, ignore_index=True)
        order = ["Timestamp"] + sorted(list(set(data.columns) - {"Timestamp"}))
        data = data.reindex(order, axis=1)
        data = data.sort_values(by=sort_by, ascending=False)
        table = Table(title=self.__repr__())
        for col in data.columns:
            table.add_column(col, style="cyan", justify="right")
        for _, row in data.iterrows():
            table.add_row(*map(str, [*row]))
        console.log(table)
