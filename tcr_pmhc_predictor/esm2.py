import torch
from transformers import EsmModel, EsmTokenizer


class ESM2Encoder:
    def __init__(self, model_dir: str, device: torch.device, use_fp16: bool = True):
        self._model_dir = model_dir
        self._device = device
        self._use_fp16 = use_fp16
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        self._tokenizer = EsmTokenizer.from_pretrained(self._model_dir)
        self._model = EsmModel.from_pretrained(self._model_dir)
        if self._use_fp16:
            self._model = self._model.half()
        self._model.to(self._device)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    @property
    def model(self):
        self._load()
        return self._model

    def extract_embeddings(
        self,
        sequences: list[str],
        max_length: int,
        d_model: int = 1280,
        batch_size: int = 256,
    ) -> torch.Tensor:
        self._load()
        all_embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            encoded = self._tokenizer(
                batch_seqs,
                padding="max_length",
                max_length=max_length + 2,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self._device)
            attention_mask = encoded["attention_mask"].to(self._device)

            with torch.no_grad():
                outputs = self._model(input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state[:, 1:-1, :]

            all_embeddings.append(hidden.reshape(-1, max_length * d_model).float().cpu())

        return torch.cat(all_embeddings, dim=0)

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            self._tokenizer = None
            torch.cuda.empty_cache()
