##############################################################
### R6 class to wrap C++ NN class and let user add R methods
### -- C++ NN methods exposed here
### -- new R methods found in additional_parts.R
##############################################################

NN_R <- R6::R6Class(
	public = list(

		Cpp = NULL,

		initialize = function(){
			self$Cpp = new("NN")
		},

		##############################################################
		### EXPOSE C++ NN METHODS THAT ADD LAYERS
		##############################################################

		add_layer = function(name, size, optional_parameter = NA){
			if(is.na(optional_parameter)){
				self$Cpp$add_layer(name, size)
			}else{
				self$Cpp$add_layer(name, size, optional_parameter)
			}
		},

		add_compute_layer = function(size, activation = "linear",
									 output_layer = FALSE, learning_rate = 0.6,
									 use_bias = TRUE, init_bias_min = -1, init_bias_max = 1){
			self$Cpp$add_compute_layer(size, activation,
									   output_layer, learning_rate,
									   use_bias, init_bias_min, init_bias_max)
		},

		##############################################################
		### EXPOSE C++ NN METHODS THAT MANAGE CONNECTIONS
		##############################################################

		add_connection_set = function(name, optional_parameter = NA){
			if(is.na(optional_parameter)){
				self$Cpp$add_connection_set(name)
			}else{
				self$Cpp$add_connection_set(name, optional_parameter)
			}
		},

		create_connections_in_sets = function(min_random_weight = -1, max_random_weight = 1){
			self$Cpp$create_connections_in_sets(min_random_weight, max_random_weight)
		},

		##############################################################
		### EXPOSE C++ NN GETTERS AND SETTERS
		##############################################################

		# expose input / output getters and setters
		get_input_at = function(pos) self$Cpp$get_input_at(pos),
		set_input_at = function(pos, data_in) self$Cpp$set_input_at(pos, data_in),
		input_at = function(pos, data_in) self$Cpp$input_at(pos, data_in),
		get_output_from = function(pos) self$Cpp$get_output_from(pos),
		get_output_at = function(pos) self$Cpp$get_output_at(pos),
		set_output_at = function(pos, data_in) self$Cpp$set_output_at(pos, data_in),

		# expose encode / recall methods
		encode_all = function(fwd = TRUE) self$Cpp$encode_all(fwd),
		recall_all = function(fwd = TRUE) self$Cpp$recall_all(fwd),

		# expose weights getters and setters
		get_weights_at = function(pos) self$Cpp$get_weights_at(pos),
		get_weight_at = function(pos, connection) self$Cpp$get_weight_at(pos, connection),
		set_weights_at = function(pos, data_in) self$Cpp$set_weights_at(pos, data_in),
		set_weight_at = function(pos, connection, value) self$Cpp$set_weight_at(pos, connection, value),

		# expose bias getters and setters
		get_biases_at = function(pos) self$Cpp$get_biases_at(pos),
		get_bias_at = function(pos, pe) self$Cpp$get_bias_at(pos, pe),
		set_biases_at = function(pos, data_in) self$Cpp$set_biases_at(pos, data_in),
		set_bias_at = function(pos, pe, value) self$Cpp$set_bias_at(pos, pe, value),

		# expose misc getters and setters
		get_misc_values_at = function(pos) self$Cpp$get_misc_values_at(pos),
		set_misc_values_at = function(pos, data_in) self$Cpp$set_misc_values_at(pos, data_in),

		##############################################################
		### EXPOSE C++ NN PRINT METHODS
		##############################################################

		# expose print methods
		component_ids = function() self$Cpp$component_ids(),
		size = function() self$Cpp$size(),
		sizes = function() self$Cpp$sizes(),
		print = function() self$Cpp$print(),
		show = function() self$Cpp$show(),
		outline = function() self$Cpp$outline(),
		get_topology_info = function() self$Cpp$get_topology_info()
	)
)
