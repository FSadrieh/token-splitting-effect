from transformers import DataCollatorForWholeWordMask
import torch

class CustomDataCollator(DataCollatorForWholeWordMask):
    """We use the DataCollatorForLanguageModeling from transformers but do not want to pad the input_ids. Therefore, this class overrides the torch_call function."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)

    def torch_call(self, examples: list):
        """The torch_call function gets:
            examples: [{"input_ids": [], "attention_mask": [], "special_tokens_mask": []}], where the outer array is the batch
        It converts these examples to a batch with a masked token at the end."""

        scalar_labels = torch.stack([torch.tensor(example["label"]) for example in examples])
        input_ids = torch.stack([torch.tensor(example["input_ids"]) for example in examples])
        attention_mask = torch.stack([torch.tensor(example["attention_mask"]) for example in examples])
        
        labels = torch.ones_like(input_ids) * -100

        for i in range(input_ids.shape[0]):
            if input_ids[i][-1] != self.tokenizer.pad_token_id:
                input_ids[i][-1] = self.tokenizer.mask_token_id
                labels[i][-1] = scalar_labels[i]
            else:
                pad_index = torch.where(input_ids[i] == self.tokenizer.pad_token_id)[0][0] #To get the first pad token index
                input_ids[i][pad_index] = self.tokenizer.mask_token_id
                labels[i][pad_index] = scalar_labels[i]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

