import pickle as pkl
import typing as tp


def loadPickle(path: str) -> tp.Any:

    data: tp.Any
    with open(path, "rb") as fid:
        data = pkl.load(fid)

    return data


def savePickle(path: str, data: tp.Any) -> None:

    with open(path, "wb") as fid:
        pkl.dump(data, fid)
