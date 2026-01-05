from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    BatchEncoding,
)
from ai_training_sge_data.data_loaders import loadDataForEmbeddingExtraction
import torch
import sys
import pandas as pd
import numpy as np
import numpy.typing as npt


def meanEmbedding(
    raw_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:

    _raw_embeddings: npt.NDArray[np.float32] = raw_embeddings.detach().numpy()
    _attention_mask: npt.NDArray[np.float32] = (
        attention_mask.to(torch.float32).detach().numpy()
    )
    prod: npt.NDArray[np.float32] = np.einsum(
        "ijk,ij->ijk", _raw_embeddings, _attention_mask
    )
    prod = prod.sum(axis=1) / _attention_mask.sum(axis=1, keepdims=True)
    return torch.tensor(prod, dtype=torch.float32)


def extractEmbeddings(
    sequences: pd.Series,
    tokeniser: PreTrainedTokenizer,
    model: PreTrainedModel,
    remove_cls_token: bool = True,
) -> torch.Tensor:

    tokenized_input: BatchEncoding = tokeniser(
        text=sequences.tolist(), return_attention_mask=True, return_tensors="pt"
    )
    input_ids: torch.Tensor = tokenized_input["input_ids"]
    attention_mask: torch.Tensor = tokenized_input["attention_mask"]

    torch_out: torch.Tensor = model(
        input_ids,
        attention_mask=attention_mask,
        encoder_attention_mask=attention_mask,
        output_hidden_states=True,
    ).hidden_states[-1]

    if remove_cls_token:
        torch_out = torch_out[..., 1:, :]
        attention_mask = attention_mask[..., 1:]

    torch_out = meanEmbedding(torch_out, attention_mask)

    return torch_out


if __name__ == "__main__":

    data_file: str = sys.argv[1]

    model_key: str = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    tokeniser_key: str = model_key

    """
    Other model keys:
    InstaDeepAI/nucleotide-transformer-2.5b-1000g
    InstaDeepAI/nucleotide-transformer-2.5b-multi-species
    InstaDeepAI/nucleotide-transformer-500m-1000g
    InstaDeepAI/nucleotide-transformer-500m-human-ref
    InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
    InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
    InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
    InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species
    InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
    """

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        tokeniser_key, trust_remote_code=True
    )
    model: PreTrainedModel = AutoModelForMaskedLM.from_pretrained(
        model_key, trust_remote_code=True
    )

    data: pd.DataFrame = loadDataForEmbeddingExtraction(data_path=data_file)
    alt_embeddings: torch.Tensor = extractEmbeddings(
        data.Sequence[:10], tokenizer, model
    )
    ref_embeddings: torch.Tensor = extractEmbeddings(
        data.pam_seq[:10], tokenizer, model
    )

    variant_embeddings: torch.Tensor = alt_embeddings - ref_embeddings
    print(variant_embeddings)
