[
  {
    num_args: 1,
    num_variables: 0,
    generate_config(): {
      type : "Substitution",
      format : ["num"],
    },
  },
  {
    num_args: 1,
    num_variables: 1,
    generate_config(i): {
      type : "Substitution",
      format : [i],
    },
  },
  {
    num_args: 2,
    num_variables: 0,
    generate_config(): {
      type : ["Add", "Sub"],
      format : [["num", "num"]],
      commutative: true,
    },
  },
  {
    num_args: 2,
    num_variables: 1,
    generate_config(i): {
      type : ["Add", "Sub"],
      format : [[i, "num"]],
      commutative: true,
    },
  },

  // {
  //   num_args: 2,
  //   num_variables: 2,
  //   generate_config(i, j): {
  //     type : ["Add", "Sub", "Max", "Min"],
  //     format : [[i, j]],
  //     commutative: true,
  //   },
  // },
]
