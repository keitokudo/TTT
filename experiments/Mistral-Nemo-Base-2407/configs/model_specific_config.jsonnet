
{
  model_name_or_path: "/work/pretrained_lms/mistralai/Mistral-Nemo-Base-2407",
  tokenizer_name_or_path: self.model_name_or_path,
  model_max_length: 4096,
  
  Decoding: {
    layer_save_step: 4,
    eos_token_id: 1010,
    pad_token_id: 2,
  },
  
  Datasets: {
    batch_size: 8,
  },
}
