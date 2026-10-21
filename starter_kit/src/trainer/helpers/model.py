from typing import Union, Dict, Any, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import numpy as np


class Model:
    def __init__(self, params: Dict[str, Union[str, int, float, bool]]) -> None:

        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=str(params["model_path"])
        )

        if params["dropout"] is not None:
            # https://huggingface.co/docs/transformers/model_doc/modernbert#transformers.ModernBertConfig.classifier_dropout
            # https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Config.classifier_dropout
            # https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartConfig.classifier_dropout
            # https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaConfig.classifier_dropout
            # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertConfig.classifier_dropout
            if hasattr(self.config, "classifier_dropout"):
                self.config.classifier_dropout = float(params["dropout"])
            # https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1031
            elif hasattr(self.config, "pooler_dropout") and not hasattr(self.config, "cls_dropout"):
                self.config.cls_dropout = float(params["dropout"])
                self.config.problem_type = "single_label_classification"
            # https://huggingface.co/docs/transformers/model_doc/albert#transformers.AlbertConfig.classifier_dropout_prob
            elif hasattr(self.config, "classifier_dropout_prob"):
                self.config.classifier_dropout_prob = float(params["dropout"])
            else:
                # https://discuss.huggingface.co/t/classifier-dropout-for-decodermodel-forsequenceclassification-classes/113989
                # https://huggingface.co/docs/transformers/model_doc/gpt_neox#transformers.GPTNeoXConfig.classifier_dropout
                # GPT2, OPT
                print("Warning: classifier_dropout attribute not found in AutoConfig. Dropout will make no effect.")

        # The checkpoint's own head size, read before the override below. A
        # 3-way NLI checkpoint (mDeBERTa, XLM-R-xnli) does not match our 4-way
        # task, and transformers>=5 raises on that mismatch instead of warning.
        checkpoint_num_labels = getattr(self.config, "num_labels", None)
        self.config.num_labels = int(params["num_labels"])

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=str(params["tokenizer_path"])
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=str(params["model_path"]),
            config=self.config,
            ignore_mismatched_sizes=(
                checkpoint_num_labels is not None
                and int(checkpoint_num_labels) != int(params["num_labels"])
            ),
        )
        # transformers>=5 loads with dtype="auto", which honours the checkpoint's
        # torch_dtype. mDeBERTa's config says float16, and this trainer has no
        # autocast or GradScaler, so force fp32 before the optimizer sees it.
        self.model = self.model.float()

        # Optional partial unfreezing: freeze bottom N encoder layers
        if params.get("freeze_bottom_layers") is not None:
            try:
                n_freeze = int(params["freeze_bottom_layers"])
            except (ValueError, TypeError):
                n_freeze = None
            if n_freeze is not None:
                mod = self.model
                layers_list = None
                if hasattr(mod, "base_model"):
                    bm = mod.base_model
                    if hasattr(bm, "layers"):
                        layers_list = bm.layers
                    elif hasattr(bm, "encoder") and hasattr(bm.encoder, "layer"):
                        layers_list = bm.encoder.layer
                elif hasattr(mod, "encoder"):
                    enc = mod.encoder
                    if hasattr(enc, "layer"):
                        layers_list = enc.layer
                    elif hasattr(enc, "layers"):
                        layers_list = enc.layers
                elif hasattr(mod, "transformer"):
                    tr = mod.transformer
                    if hasattr(tr, "h"):
                        layers_list = tr.h
                    elif hasattr(tr, "layers"):
                        layers_list = tr.layers
                elif hasattr(mod, "layers"):
                    layers_list = mod.layers
                if layers_list is None:
                    print(f"Warning: freeze_bottom_layers={n_freeze} requested but encoder layer list not found; skipping freeze.")
                else:
                    total = len(layers_list)
                    if n_freeze <= 0:
                        # nothing frozen
                        for layer in layers_list:
                            for p in layer.parameters():
                                p.requires_grad = True
                        if hasattr(mod, "classifier"):
                            for p in mod.classifier.parameters():
                                p.requires_grad = True
                        print("freeze_bottom_layers <= 0; all encoder+classifier parameters trainable.")
                    elif n_freeze >= total:
                        for layer in layers_list:
                            for p in layer.parameters():
                                p.requires_grad = False
                        if hasattr(mod, "classifier"):
                            for p in mod.classifier.parameters():
                                p.requires_grad = True
                        print(f"freeze_bottom_layers={n_freeze} >= total layers ({total}); frozen all encoder layers, classifier trainable.")
                    else:
                        for i, layer in enumerate(layers_list):
                            for p in layer.parameters():
                                p.requires_grad = (i >= n_freeze)
                        if hasattr(mod, "classifier"):
                            for p in mod.classifier.parameters():
                                p.requires_grad = True
                        print(f"Partial unfreezing: frozen bottom {n_freeze}/{total} encoder layers; classifier trainable.")

        # For pythia
        if not self.tokenizer.pad_token:
            print("No padding token detected. Expanding tokenizer and setting padding token to [PAD].")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Optional embedding freeze. Kept after resize_token_embeddings so a
        # freshly grown embedding table is frozen too.
        if params.get("freeze_embeddings"):
            for p in self.model.get_input_embeddings().parameters():
                p.requires_grad = False
            print("Input embeddings frozen.")

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.1%})")

        self.emb_size: int = self.model.config.hidden_size
        # self.model.config.max_position_embeddings
        self.max_length: int = int(params["max_length"])

    def forward(self, batch: Dict[str, torch.Tensor], device: torch.device) -> SequenceClassifierOutput:
        inputs: Dict[str, torch.Tensor] = {
            "input_ids": batch["input_ids"].to(device=device),
            "attention_mask": batch["attention_mask"].to(device=device),
        }
        
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"].to(device=device)

        # Check if labels are provided in the batch
        if "labels" in batch:
            inputs["labels"] = batch["labels"].to(device=device)

        return self.model(**inputs)

    def tokenize(self, data: np.ndarray) -> dict[str, Any]:
        texts_first, texts_second = zip(*data)
        texts_first: np.ndarray
        texts_second: np.ndarray

        # A whole split is tokenised in a single call, so "longest" pads every
        # row to the longest pair in that split (34 tokens on track 1) instead
        # of to max_length (64). max_length still caps truncation. Same logits,
        # roughly half the attention work.
        encs: BatchEncoding = self.tokenizer(
            list(texts_first),
            list(texts_second),
            padding="longest",
            truncation="only_second",
            return_tensors="pt",
            max_length=self.max_length,
        )

        return encs.data
    
    def to(self, device: torch.device) -> None:
        self.model.to(device=device)

    def parameters(self) -> torch.nn.Parameter:
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def decode(self, token_ids: List[List[int]]) -> List[str]:
        decoded_outputs:List[str] = self.tokenizer.batch_decode(sequences=token_ids, skip_special_tokens=False)
        return [text.replace(self.tokenizer.pad_token, "") for text in decoded_outputs]

    def save(self, path: str) -> None:
        self.model.save_pretrained(save_directory=path)
        self.tokenizer.save_pretrained(save_directory=path)
