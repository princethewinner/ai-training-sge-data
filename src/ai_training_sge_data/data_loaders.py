import typing as tp
from copy import deepcopy
from ai_training_sge_data.io import savePickle, loadPickle
import pandas as pd
from dataclasses import dataclass, field
import warnings

_T = tp.TypeVar("_T", bound="BaseDataFrame")


class BaseDataFrame:
    def __init__(self) -> None:
        pass

    def serialise(self) -> tp.Dict[str, tp.Any]:
        return deepcopy(vars(self))

    def save(self, path: str) -> None:
        path = str(path)
        serializaed_object: tp.Dict[str, tp.Any] = self.serialise()
        savePickle(path, serializaed_object)

    @classmethod
    def load(
        cls: tp.Type[_T],
        _from: tp.Union[str, tp.Dict[str, tp.Any]],
    ) -> _T:
        _content: tp.Dict[str, tp.Any]
        if isinstance(_from, dict):
            _content = _from
        else:
            path = str(_from)
            _content = loadPickle(path)

        return cls(**_content)


@dataclass
class GeneDataFrame(BaseDataFrame):
    gene_name: str
    task_type: str
    sge_data: pd.DataFrame = field(repr=False)
    ref_data: tp.Optional[pd.DataFrame] = field(repr=False, default=None)
    gene_metadata: tp.Optional[tp.Dict[str, str]] = field(repr=False, default=None)
    n_samples: int = field(repr=True, default=-1)
    info: str = field(repr=True, default="No coordinates in the dataframe.")

    def _updateSampleInfo(self) -> None:
        self.n_samples = self.sge_data.shape[0]

    def __post_init__(self) -> None:
        self._updateSampleInfo()


@dataclass
class SGEDataFrame(BaseDataFrame):
    holder: tp.Dict[str, GeneDataFrame]
    gene_view_order: tp.List[str] = field(repr=False, default_factory=list)

    def __post_init__(self) -> None:
        self.gene_view_order = list(self.holder.keys())

    def __repr__(self) -> str:
        content: str = "\n\t".join(
            [f"{gene_name}={holder}" for gene_name, holder in self.holder.items()]
        )
        return f"{self.__class__.__name__}():\n\t{content}"

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, gene_name: str) -> GeneDataFrame:
        if gene_name not in self.holder:
            raise KeyError()  # @TODO: Add message
        return self.holder[gene_name]

    def combineSGEData(
        self, gene_list: tp.List[str]
    ) -> tp.Tuple[pd.DataFrame, tp.Dict[str, int]]:
        df_list: tp.List[pd.DataFrame] = []
        gene_start_index: tp.Dict[str, int] = dict()
        run_index: int = 0
        for key in gene_list:
            df_col: pd.DataFrame = self[key].sge_data.copy()
            df_col["gene"] = key
            df_list.append(df_col)
            gene_start_index[key] = run_index
            run_index += df_col.shape[0]

        return pd.concat(df_list, axis=0).reset_index(drop=True), gene_start_index

    @classmethod
    def load(
        cls: tp.Type[_T],
        _from: tp.Union[str, tp.Dict[str, tp.Any]],
    ) -> _T:
        _content: tp.Dict[str, tp.Any]
        if isinstance(_from, dict):
            _content = _from
        else:
            path = str(_from)
            _content = loadPickle(path)

        for key in _content["holder"]:
            _content["holder"][key] = GeneDataFrame.load(_content["holder"][key])

        return cls(**_content)

    @property
    def supported_gene_list(self) -> tp.List[str]:
        return self.gene_view_order

    @property
    def combined_sge_dataframe_for_gene(self) -> pd.DataFrame:
        return self.combineSGEData(self.supported_gene_list)[0]

    @property
    def combined_dataframe_for_valiant(self) -> pd.DataFrame:
        df_list: tp.List[pd.DataFrame] = []
        for key in self.supported_gene_list:
            if self[key].ref_data is not None:
                df_col: pd.DataFrame = self[key].ref_data.copy()  # type: ignore
                df_col["gene"] = key
                df_list.append(df_col)

        if len(df_list) > 0:
            return pd.concat(df_list, axis=0).reset_index(drop=True)

        warnings.warn("None of the genes has valiant files.", UserWarning)
        return pd.DataFrame()

    @property
    def combined_metadata(self) -> tp.Dict[str, tp.Any]:
        metadata_dict: tp.Dict[str, tp.Any] = dict()
        for key in self.supported_gene_list:
            metadata_dict[key] = self[key].gene_metadata
        return metadata_dict
