import errno
import glob
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import datasets
import lightning as L
import torch
from print_on_steroids import logger
from torch.utils.data.dataloader import DataLoader
from transformers import PreTrainedTokenizerFast

from src.training.custom_data_collator import CustomDataCollator

from dlib.frameworks.pytorch import get_rank

if TYPE_CHECKING:
    from train import TrainingArgs


# Define a Lightning Data Module for Language Modeling
class LMDataModule(L.LightningDataModule):
    def __init__(
        self,
        training_args: "TrainingArgs",
        tokenizer: PreTrainedTokenizerFast,
        prompt_length: int,
    ):
        # Initialize the data module with training arguments, tokenizer, and prompt length
        super().__init__()
        self.args = training_args
        self.data_dir = training_args.data_dir
        train_file, val_file = (
            self.data_dir / self.args.train_file,
            self.data_dir / self.args.val_file,
        )

        logger.debug(f"Train file path: {train_file} val file path: {val_file}")

        self.train_file = str(train_file)
        self.val_file = str(val_file)
        self.tokenizer_path = self.args.tokenizer_path or self.args.hf_model_names[0]
        self.local_rank = get_rank()

        # Initialize tokenizer and configure max token length for the dataset
        self.tokenizer = tokenizer
        self.tokenize_function = None
        self.max_length = self.args.block_size - prompt_length

    def prepare_data(self) -> None:
        # Prepare data by checking cache and processing datasets if necessary
        cache_exists, cache_path = self._get_dataset_cache_path(self.tokenizer_path)
        if not cache_exists:
            logger.info(f"Could not find cached processed dataset: {cache_path}, creating it now...")
            # Load and process dataset if not cached
            processed_datasets = self.load_and_process_dataset(self.tokenizer, str(self.data_dir / "tokenized"))
            logger.info(f"Saving dataset to {cache_path}...")
            processed_datasets.save_to_disk(cache_path, num_proc=self.args.preprocessing_workers)
        else:
            logger.success(f"Found cached processed dataset: {cache_path}.")
        # Exit if only data preprocessing is needed
        if self.args.data_preprocessing_only:
            exit(0)

    def setup(self, stage):
        # Setup datasets for training or validation stage
        cache_exists, cache_path = self._get_dataset_cache_path(self.tokenizer_path)
        assert (
            cache_exists
        ), f"Could not find cached processed dataset: {cache_path}, should have been created in prepare_data()"

        logger.info(f"Loading cached processed dataset from {cache_path}...", rank0_only=False)
        processed_datasets = datasets.load_from_disk(cache_path)

        # Initialize data collator for batching and padding
        data_collator = CustomDataCollator(self.tokenizer)
        #data_collator = DefaultDataCollator()

        # Assign datasets and data collator for training and validation
        self.train_dataset = processed_datasets["train"]
        self.val_dataset = processed_datasets["val"]
        self.data_collator = data_collator

    def load_and_process_dataset(self, tokenizer, tokenized_data_dir):
        # Determine the file format of the dataset (txt, jsonl, etc.)
        extension = self.train_file.split(".")[-1]
        if extension in ("txt", "raw"):
            extension = "text"
        elif extension == "jsonl":
            extension = "json"

        # Define paths for training and validation data files
        data_files = {"train": self.train_file, "val": self.val_file}

        logger.info("Loading raw dataset...")
        
        # Create temporary directory for dataset caching, if disk space conservation is enabled
        tmp_load_dataset_cache_dir = tempfile.mkdtemp(dir=tokenized_data_dir) if self.args.conserve_disk_space else None
        train_val_datasets = datasets.load_dataset(
            extension,
            data_files=data_files,
            name=str(self.data_dir).replace("/", "_"),
            num_proc=self.args.preprocessing_workers,
            cache_dir=tmp_load_dataset_cache_dir,
        )

        # Debug logging for the first two samples of the training dataset
        if self.local_rank == 0:
            logger.debug((train_val_datasets, train_val_datasets["train"][:2]))

        # Disable dataset caching if disk space conservation is active
        if self.args.conserve_disk_space:
            datasets.fingerprint.disable_caching()

        # Process the datasets in chunks using the tokenizer
        processed_datasets = self.process_dataset_in_chunks(tokenizer=tokenizer, train_val_datasets=train_val_datasets)

        # processed_datasets["train"] = processed_datasets["train"].shuffle(seed=self.args.seed) # <-- this is bad, triggers super expensive .flatten_indices op when .save_to_disk
        logger.success(
            f"Rank {self.local_rank} | Finished processing datasets: {processed_datasets} | First sample len: {len(processed_datasets['train'][0]['input_ids'])}"
        )

        # Clean up the dataset loading cache directory, if it was used
        if self.args.conserve_disk_space:
            logger.info("Cleaning dataset loading cache...")
            try:
                shutil.rmtree(tmp_load_dataset_cache_dir)
            except OSError as e:
                # Reraise unless ENOENT: No such file or directory
                # (ok if directory has already been deleted)
                if e.errno != errno.ENOENT:
                    raise

            datasets.fingerprint.enable_caching()

        return processed_datasets

    def process_dataset_in_chunks(self, tokenizer, train_val_datasets):
        """Expects input data to be one document per line. Tokenizes the documents and splits into chunks of max_sequence_legth."""
        tokenized_datasets = train_val_datasets.map(
            make_tokenize_function(tokenizer, max_seq_length=self.max_length, data_dir=self.args.data_dir),
            batched=True,
            num_proc=1,  # Should use only one process to leverage tokenizers parallelism
            remove_columns=["text"],  # Remove original text column after tokenization
            load_from_cache_file=not self.args.overwrite_data_cache,
            desc="Running tokenizer on every text in dataset",
        )

        return tokenized_datasets

    def train_dataloader(self):
        # Create DataLoader for training data
        common_args = dict(
            batch_size=self.args.micro_batch_size,
            num_workers=self.args.workers,
            persistent_workers=(
                True if self.args.workers > 0 else False
            ),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            shuffle=True,  # Shuffle training data
        )
        return DataLoader(self.train_dataset, collate_fn=self.data_collator, **common_args)

    def val_dataloader(self):
        # Create DataLoader for validation data
        common_args = dict(
            batch_size=self.args.eval_micro_batch_size,
            num_workers=self.args.workers,
            persistent_workers=(
                True if self.args.workers > 0 else False
            ),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
        )
        return DataLoader(self.val_dataset, collate_fn=self.data_collator, **common_args)

    def _get_dataset_cache_path(self, tokenizer_name: str):
        # Generate the path for cached dataset based on tokenizer and other parameters
        tokenizer_name = Path(self.tokenizer_path).as_posix().replace("/", "_")
        
        # Create a tokenize function hash to ensure cache consistency
        # This is to prevent a rarely occurring bug where the hash of the tokenize function changes between runs
        self.tokenize_function = self.tokenize_function if self.tokenize_function else make_tokenize_function(self.tokenizer, max_seq_length=self.max_length, data_dir=self.args.data_dir)
        tokenize_fn_hash = datasets.fingerprint.Hasher.hash(self.tokenize_function)
        
        # Define the directory and file path for the cached data
        tokenized_data_dir = str(self.data_dir / "tokenized")
        cache_path = os.path.join(
            tokenized_data_dir,
            f"{self.args.train_file}.{self.args.val_file}.seq_len_{self.max_length}.tokenizer_{tokenizer_name}.tokenize_fn_hash_{tokenize_fn_hash}.arrow",
        )
        
        # Check for existing cache files that might match
        maybe_cache_path = os.path.join(
            tokenized_data_dir,
            f"{self.args.train_file}.{self.args.val_file}.seq_len_{self.max_length}.tokenizer_{tokenizer_name}.tokenize_fn_hash_.*.arrow",
        )
        maybe_cache_path_match_list = glob.glob(maybe_cache_path)

        # Determine if a valid cache file exists and return its path
        if os.path.exists(cache_path):
            return True, cache_path
        elif len(maybe_cache_path_match_list) > 0 and os.path.exists(maybe_cache_path_match_list[0]):
            logger.warning(
                f"Rank {self.local_rank} | Did not find cached processed dataset: {cache_path} but {maybe_cache_path_match_list[0]}.",
                "The tokenize function hash can change with small, functionally meaningless code changes in the tokenizers library.",
                "Proceeding with existing found cache.",
            )
            return True, maybe_cache_path_match_list[0]
        else:
            return False, cache_path


def make_tokenize_function(tokenizer, max_seq_length, data_dir):
    """Needs to be outside of DataModule because of hashing error in dataset.map"""

    # Define a tokenize function for processing text data
    def tokenize_function(examples):
        tokenized =  tokenizer(
            examples["text"],
            max_length=max_seq_length,
            padding=True,
            truncation=True,
        )
        
        if "emotion" in data_dir.name:
            scalar_labels = torch.tensor([tokenizer("sadness", add_special_tokens=False)["input_ids"][0] if x == 0 else tokenizer("joy", add_special_tokens=False)["input_ids"][0] if x == 1 else tokenizer("love", add_special_tokens=False)["input_ids"][0] if x == 2 else tokenizer("anger", add_special_tokens=False)["input_ids"][0] if x == 3 else tokenizer("fear", add_special_tokens=False)["input_ids"][0] if x == 4 else tokenizer("surprise", add_special_tokens=False)["input_ids"][0] for x in examples["label"]])
        else:
            scalar_labels = torch.tensor([tokenizer("negative", add_special_tokens=False)["input_ids"][0] if x == 0 else tokenizer("positive", add_special_tokens=False)["input_ids"][0] for x in examples["label"]])

        tokenized["label"] = scalar_labels
        return tokenized

    return tokenize_function
