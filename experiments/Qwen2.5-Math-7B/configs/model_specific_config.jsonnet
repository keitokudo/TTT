{
  model_name_or_path: "/work/pretrained_lms/Qwen/Qwen2.5-Math-7B",
  tokenizer_name_or_path: self.model_name_or_path,
  model_max_length: 4096,
  
  Decoding: {
    layer_save_step: 4,
    eos_token_id: 198,
    pad_token_id: 151643,
  },
  
  Datasets: {
    batch_size: 4,
  },
}
