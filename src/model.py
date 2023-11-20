import lightning as L
import torch
from print_on_steroids import logger
from torch.optim import AdamW
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.optimization import get_scheduler
from warmup_scheduler import GradualWarmupScheduler


# Define a basic Language Model using PyTorch Lightning
class BasicLM(L.LightningModule):
    def __init__(
        self,
        model_name_or_path_1: str,
        from_scratch: bool,
        learning_rate: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        lr_schedule: str,
        warmup_period: int,
        eval_interval: int,
        prompt_length: int,
        model_name_or_path_2: str | None = None,
        epsilon: float = 1e-8,
        save_hyperparameters: bool = True,
        local_adapter: str | None = None,
    ) -> None:
        # Initialize the LightningModule
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters(ignore=["save_hyperparameters"])
        
        # Load configuration and initialize the first transformer model
        config_1 = AutoConfig.from_pretrained(model_name_or_path_1, return_dict=True)
        self.model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(model_name_or_path_1, config=config_1)
            if not from_scratch
            else AutoModelForSequenceClassification.from_config(config=config_1)
        )

        # Optionally load a second transformer model, if provided
        if model_name_or_path_2:
            config_2 = AutoConfig.from_pretrained(model_name_or_path_2, return_dict=True)
            self.model_2: PreTrainedModel = (
                AutoModelForSequenceClassification.from_pretrained(model_name_or_path_2, config=config_2)
                if not from_scratch
                else AutoModelForSequenceClassification.from_config(config=config_2)
            )
            for param in self.model_2.parameters():
                param.requires_grad = False
        else:
            self.model_2 = None

        # Freeze parameters of the model(s) for prompt-based tuning
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize a prompt embedding layer
        embedding_size = self.model.config.hidden_size
        self.soft_prompt = torch.nn.Embedding(prompt_length, embedding_size)
        self.prompt_tokens = torch.arange(prompt_length).long()

        # Store optimization parameters
        prompt_token_weights = self.model.get_input_embeddings()(self.prompt_tokens).detach().clone()
        self.soft_prompt.weight = torch.nn.Parameter(prompt_token_weights.to(torch.float32))

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr_schedule = lr_schedule
        self.warmup_period = warmup_period
        self.eval_interval = eval_interval
        self.epsilon = epsilon


    def forward(self, input_ids, attention_mask, labels, token_type_ids=None):
        # Forward pass through the model
        embedded_input = self.model.get_input_embeddings()(input_ids)
        prompt = self.soft_prompt(self.prompt_tokens.to(self.device)).unsqueeze(0).expand(embedded_input.shape[0], -1, -1)

        # Generate prompt embeddings and concatenate with input embeddings
        inputs_embeds = torch.cat([prompt, embedded_input], dim=1)
        
        # Adjust attention mask for the concatenated prompt
        attention_mask = torch.cat([torch.ones(prompt.shape[0], prompt.shape[1]).to(self.device), attention_mask], dim=1)

        loss = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss
        #loss_2 = self.model_2(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss
        #loss = (loss_1 + loss_2) / 2
        return loss

    def training_step(self, batch, batch_idx):
        # Perform a training step
        loss = self(**batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Perform a validation step
        loss = self(**batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        # Configure the optimizers and learning rate schedulers
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.learning_rate}, weight decay: {self.weight_decay} and warmup steps: {self.warmup_period}"
            )

        named_parameters = list(self.model.named_parameters())

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
            scheduler_config = {"frequency": self.eval_interval, "monitor": "train/loss"}
        else:
            scheduler_name = self.lr_schedule
            if scheduler_name == "constant" and self.warmup_period > 0:
                scheduler_name += "_with_warmup"
            scheduler = get_scheduler(
                scheduler_name,
                optimizer,
                num_warmup_steps=int(self.warmup_period),
                num_training_steps=self.trainer.max_steps,
            )
            scheduler_config = {"frequency": 1}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", **scheduler_config},
        }
