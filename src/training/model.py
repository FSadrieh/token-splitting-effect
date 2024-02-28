import lightning as L
import torch
import math
from print_on_steroids import logger
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForMaskedLM
from transformers.optimization import get_scheduler
from warmup_scheduler import GradualWarmupScheduler

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from similarity_calculation import calculate_sim  # noqa: E402


# Define a basic Language Model using PyTorch Lightning
class BasicLM(L.LightningModule):
    def __init__(
        self,
        model_names_or_paths: list[str],
        tokenizer: PreTrainedTokenizerFast,
        from_scratch: bool,
        learning_rate: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        lr_schedule: str,
        warmup_period: int,
        prompt_length: int,
        init_embedding_mode: str,
        epsilon: float = 1e-8,
        init_seed: int = 42,
        save_hyperparameters: bool = True,
        local_soft_prompt: str = None,
        init_text: str | None = None,
        init_embedding_models: str | None = None,
    ) -> None:
        # Initialize the LightningModule
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters(ignore=["save_hyperparameters"])

        self.models = []
        for model_name_or_path in model_names_or_paths:
            if model_name_or_path is None:
                raise ValueError("model_name_or_path cannot be None")

            # Load configuration and initialize the first transformer model
            config = AutoConfig.from_pretrained(model_name_or_path, return_dict=True)
            model: PreTrainedModel = (
                AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)
                if not from_scratch
                else AutoModelForMaskedLM.from_config(config=config)
            )

            # Since we only want to train the SPs we freeze all model parameters
            for param in model.parameters():
                param.requires_grad = False

            self.models.append(model)

        # For some functions we need a concrete model, so we just take the first one
        self.model = self.models[0]

        # Initialize a prompt embedding layer
        # This snippet is inspired by https://github.com/huggingface/peft/blob/main/src/peft/tuners/prompt_tuning/model.py
        embedding_size = self.model.config.hidden_size
        self.soft_prompt = torch.nn.Embedding(prompt_length, embedding_size)
        self.prompt_tokens = torch.arange(prompt_length).long()

        if local_soft_prompt:
            self.soft_prompt.load_state_dict(torch.load(local_soft_prompt))
        else:
            # Init text was not used in the paper, but has been shown to perform well in some cases. We do not recommend using it!
            if init_text:
                init_ids = tokenizer(init_text, add_special_tokens=False)["input_ids"]
                if len(init_ids) > prompt_length:
                    init_ids = init_ids[:prompt_length]
                elif len(init_ids) < prompt_length:
                    num_reps = math.ceil(prompt_length / len(init_ids))
                    init_ids = init_ids * num_reps
                init_ids = init_ids[:prompt_length]
                if init_embedding_models and init_embedding_mode != "normal":
                    init_embeddings = []
                    for init_embedding_model in init_embedding_models.split(","):
                        init_model = AutoModelForMaskedLM.from_pretrained(init_embedding_model)
                        init_embeddings.append(init_model.get_input_embeddings())

                    if init_embedding_mode == "average":
                        prompt_token_weights = torch.mean(
                            torch.stack(
                                [embedding(torch.LongTensor(init_ids)).detach().clone() for embedding in init_embeddings]
                            ),
                            dim=0,
                        )
                    elif init_embedding_mode == "mix":
                        parts = len(init_embeddings)
                        part_length = prompt_length // parts
                        init_parts = []
                        for part in range(parts - 1):
                            init_parts.append(init_ids[part * part_length : (part + 1) * part_length])
                        init_parts.append(init_ids[(parts - 1) * part_length :])
                        prompt_token_weights = torch.cat(
                            [
                                embedding(torch.LongTensor(init_part)).detach().clone()
                                for init_part, embedding in zip(init_parts, init_embeddings)
                            ],
                            dim=0,
                        )
                else:
                    init_model = (
                        AutoModelForMaskedLM.from_pretrained(init_embedding_models.split(",")[0])
                        if init_embedding_models
                        else self.model
                    )
                    embedding = init_model.get_input_embeddings()
                    prompt_token_weights = embedding(torch.LongTensor(init_ids)).detach().clone()
                self.soft_prompt.weight = torch.nn.Parameter(prompt_token_weights.to(torch.float32))

            else:
                # This was the default in the paper
                seed = torch.random.get_rng_state()
                self.soft_prompt.weight = torch.nn.Parameter(
                    torch.randn(prompt_length, embedding_size, generator=torch.random.manual_seed(init_seed)).to(torch.float32)
                )
                torch.random.set_rng_state(seed)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr_schedule = lr_schedule
        self.warmup_period = warmup_period
        self.epsilon = epsilon
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

        # We save the initial soft prompt to calculate the distance to it and to save it in the end
        self.init_soft_prompt = self.soft_prompt(self.prompt_tokens).detach().clone()

    def set_soft_prompt_weight(self, soft_prompt_weight: torch.nn.Parameter):
        self.soft_prompt.weight = soft_prompt_weight

    def forward(self, input_ids, attention_mask, labels, token_type_ids=None):
        # We need to broadcast the prompt to the batch size
        prompt = self.soft_prompt(self.prompt_tokens.to(self.device)).unsqueeze(0).expand(input_ids.shape[0], -1, -1)

        # Adjust attention mask for the concatenated prompt
        attention_mask = torch.cat([torch.ones(prompt.shape[0], prompt.shape[1]).to(self.device), attention_mask], dim=1)
        labels = torch.cat([torch.ones(prompt.shape[0], prompt.shape[1]).to(self.device) * -100, labels], dim=1).long()

        # For each model we calculate the loss and return them in a list
        outputs = []
        for model in self.models:
            model.to(self.device)
            embedded_input = model.get_input_embeddings()(input_ids)

            # Generate prompt embeddings and concatenate with input embeddings
            inputs_embeds = torch.cat([prompt, embedded_input], dim=1)

            outputs.append(model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels))

        return outputs

    def training_step(self, batch, batch_idx):
        # Perform a training step
        outputs = self(**batch)
        # Here we take the list of model outputs and mean the losses
        loss = torch.stack([output.loss for output in outputs]).mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Perform a validation step
        outputs = self(**batch)
        loss = torch.stack([output.loss for output in outputs]).mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        current_soft_prompt = self.soft_prompt(self.prompt_tokens.to(self.device))
        initial_soft_prompt = self.init_soft_prompt.to(self.device)
        # These are four of the six similarity metrics out of similarity_calculation.py. We use them to monitor the distance to the initial soft prompt. These results are not used in the paper.
        self.log(
            "val/distance_to_init",
            calculate_sim(current_soft_prompt, initial_soft_prompt, "euclidean", pre_averaging=False),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/distance_to_init_on_avg_prompt",
            calculate_sim(current_soft_prompt, initial_soft_prompt, "euclidean", pre_averaging=True),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/sim_to_init",
            calculate_sim(current_soft_prompt, initial_soft_prompt, "cosine", pre_averaging=False),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/sim_to_init_on_avg_prompt",
            calculate_sim(current_soft_prompt, initial_soft_prompt, "cosine", pre_averaging=True),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = torch.stack([output.loss for output in outputs]).mean()
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        # Configure the optimizers and learning rate schedulers
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.learning_rate}, weight decay: {self.weight_decay} and warmup steps: {self.warmup_period}"
            )

        named_parameters = list(self.soft_prompt.named_parameters())

        # Filter out parameters that are not optimized (requires_grad == False)
        optimized_named_parameters = [(n, p) for n, p in named_parameters if p.requires_grad]

        # Set different weight decay for certain parameters
        # Do not include LayerNorm and bias terms for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in optimized_named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in optimized_named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_parameters,
            self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,  # You can also tune this
        )

        self.trainer.fit_loop.setup_data()

        # Configure learning rate scheduler based on the selected strategy
        if self.lr_schedule == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
            if self.warmup_period > 0:  # Wrap ReduceLROnPlateau to enable LR warmup
                scheduler = GradualWarmupScheduler(
                    optimizer,
                    multiplier=1,
                    total_epoch=self.warmup_period,
                    after_scheduler=scheduler,
                )
            scheduler_config = {"frequency": 1, "monitor": "train/loss"}
        else:
            scheduler_name = self.lr_schedule
            if scheduler_name == "constant" and self.warmup_period > 0:
                scheduler_name += "_with_warmup"
            scheduler = get_scheduler(
                scheduler_name,
                optimizer,
                num_warmup_steps=int(self.warmup_period),
                num_training_steps=self.trainer.max_steps
                if self.trainer.max_steps != -1
                else int(
                    int(len(self.trainer.train_dataloader) / int(self.trainer.accumulate_grad_batches))
                    * self.trainer.max_epochs
                ),
            )
            scheduler_config = {"frequency": 1}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", **scheduler_config},
        }
