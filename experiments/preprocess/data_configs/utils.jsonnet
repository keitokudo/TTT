local datatype_to_seed_definition = {
  "train": 42,
  "valid": 43,
  "test": 44,
  "probing": 45,
};

{
  datatype_to_seed(dtype):
    if std.objectHas(datatype_to_seed_definition, dtype) then
      datatype_to_seed_definition[dtype]
    else
      0,
 
  remove_null(input_array):
    if std.length(input_array) == 0 then
      []
    else
      local top = input_array[0];
      if top == null then
	self.remove_null(input_array[1:])
      else
	[top] + self.remove_null(input_array[1:]),

  permutation_sub(input_array, copyed_array):
 
    if std.length(input_array) == 0 then
      []
    else
      local top = input_array[0];
      self.remove_null(
	[
	  if top == elem then null else [top, elem]
	  for elem in copyed_array
	]
      )
      + self.permutation_sub(
	input_array[1:],
	copyed_array,
      ),

  permutation(input_array):
    self.permutation_sub(input_array, input_array),
}

    
		
