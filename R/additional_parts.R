##############################################################
### new wrapper functions to add layers via piping
##############################################################

from_input <-
	function(self, size){

		self$Cpp$add_compute_layer(
			size = size, activation = "linear",
			output_layer = FALSE, learning_rate = 0,
			use_bias = FALSE, init_bias_min = 0, init_bias_max = 0
		)
		self
	}

connect_to <-
	function(self,
			 size, activation = "linear",
			 output_layer = FALSE, learning_rate = 0.6,
			 use_bias = TRUE, init_bias_min = -1, init_bias_max = 1){

		self$Cpp$add_connection_set( "BP" )
		self$Cpp$add_compute_layer(
			size, activation,
			output_layer, learning_rate,
			use_bias, init_bias_min, init_bias_max
		)
		self
	}

ready <-
	function(self, init_weight_min = 0, init_weight_max = 1){

		self$Cpp$create_connections_in_sets(init_weight_min, init_weight_max)
		print(self$Cpp$outline())
		self
	}

##############################################################
### new PASS_FORWARD() and PASS_BACKWARD() wrapper methods
##############################################################

NN_R$set(
	"public",
	"pass_forward",
	function(input) {

		self$input_at(1, input)
		self$recall_all(TRUE)
		return(self$get_output_at(self$size()))
	}
)

NN_R$set(
	"public",
	"pass_backward",
	function(desired_or_delta) {

		self$input_at(self$size(), desired_or_delta)
		self$encode_all(FALSE)
		return(self$get_misc_values_at(1))
	}
)
