{
  model_name_or_path: "/work/pretrained_lms/01-ai/Yi-1.5-9B",
  tokenizer_name_or_path: self.model_name_or_path,
  model_max_length: 4096,
  
  Decoding: {
    layer_save_step: 6,
    eos_token_id: 144,
    pad_token_id: 0,
  },
  
  Datasets: {
    batch_size: 4,
  },
}
