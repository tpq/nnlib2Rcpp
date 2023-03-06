##############################################################
### This example builds on the simple NN_R class example to
### show how computed differentials can be passed between
### discrete models to enable multi-task learning.
##############################################################

library(nnlib2Rcpp)

# scale data from 0 to 1 range
iris_s <- as.matrix( iris [ 1 : 4 ] )
c_min <- apply( iris_s, 2, FUN = "min" )
c_max <- apply( iris_s, 2, FUN = "max" )
c_rng <- c_max - c_min
iris_s <- sweep( iris_s, 2, FUN="-", c_min )
iris_s <- sweep( iris_s, 2, FUN="/", c_rng )

# 1-hot encode class labels
mat <- model.matrix(~.+0, iris[,5,drop=F])

# Encode iris measures to latent space
encoder <- NN_R$new()
encoder$add_layer(name = "generic", size = 4)
encoder$add_connection_set( "BP" )
encoder$add_compute_layer(size = 4,
						  activation = "sigmoid", output_layer = FALSE)
encoder$add_connection_set( "BP" )
encoder$add_compute_layer(size = 2,
						  activation = "sigmoid", output_layer = FALSE)
encoder$create_connections_in_sets (0, 1)
encoder$outline()

# Decode latent space to class label
decode_to_class <- NN_R$new()
decode_to_class$add_compute_layer(size = 2,
								  activation = "linear", output_layer = FALSE, use_bias = FALSE)
decode_to_class$add_connection_set("BP")
decode_to_class$add_compute_layer(size = 3,
								  activation = "sigmoid", output_layer = TRUE)
decode_to_class$create_connections_in_sets (0, 1)
decode_to_class$outline()

# Decode latent space to re-construct iris measures (e.g., autoencoder)
decode_to_self <- NN_R$new()
decode_to_self$add_compute_layer(size = 2,
								 activation = "linear", output_layer = FALSE, use_bias = FALSE)
decode_to_self$add_connection_set( "BP" )
decode_to_self$add_compute_layer(size = 4,
								 activation = "sigmoid", output_layer = FALSE)
decode_to_self$add_connection_set( "BP" )
decode_to_self$add_compute_layer(size = 4,
								 activation = "sigmoid", output_layer = FALSE)
decode_to_self$add_connection_set("BP")
decode_to_self$add_compute_layer(size = 4, activation = "linear", output_layer = TRUE)
decode_to_self$create_connections_in_sets (0, 1)
decode_to_self$outline()

train.x <- iris_s
train.y <- mat
epochs = 1000
for(e in 1:epochs){ # for E epochs
	if(e %% 10 == 0) print(e)
	for(r in sample(1:nrow(train.x))) # for each data case
	{
		# Get latent variable
		encoder$input_at( 1, train.x[r,] )
		encoder$recall_all(TRUE)
		latent <- encoder$get_output_at(encoder$size())

		# Fork to predict TASK 1
		decode_to_class$input_at(1, latent)
		decode_to_class$recall_all()

		# Fork to predict TASK 2
		decode_to_self$input_at(1, latent)
		decode_to_self$recall_all()

		# Back-prop from TASK 1
		decode_to_class$input_at(decode_to_class$size(), train.y[r,])
		decode_to_class$encode_all(FALSE)

		# Back-prop from TASK 2
		decode_to_self$input_at(decode_to_self$size(), train.x[r,])
		decode_to_self$encode_all(FALSE)

		# Back-prop to encoder
		d1 <- decode_to_class$Cpp$get_misc_values_at(1)
		d2 <- decode_to_self$Cpp$get_misc_values_at(1)
		encoder$input_at(encoder$size(), d1)
		encoder$encode_all(FALSE)
		encoder$input_at(encoder$size(), d2)
		encoder$encode_all(FALSE)
	}
}

z <- vector("list", nrow(train.x))
out1 <- vector("list", nrow(train.x))
out2 <- vector("list", nrow(train.x))
for(r in 1:nrow(train.x)) # for each data case
{
	# Get latent variable
	encoder$input_at( 1, train.x[r,] )
	encoder$recall_all(TRUE)
	latent <- encoder$get_output_at(encoder$size())

	# Fork to predict TASK 1
	decode_to_class$input_at(1, latent)
	decode_to_class$recall_all()

	# Fork to predict TASK 2
	decode_to_self$input_at(1, latent)
	decode_to_self$recall_all()

	# Save model output
	z[[r]] <- latent
	out1[[r]] <- decode_to_class$get_output_at(decode_to_class$size())
	out2[[r]] <- decode_to_self$get_output_at(decode_to_self$size())
}

# View latent space
z_df <- do.call("rbind", z)
plot(z_df[,1], z_df[,2])

# View class label results
out1_df <- do.call("rbind", out1)
plot(out1_df[,1])
plot(out1_df[,2])
plot(out1_df[,3])

# View auto-encoder results
out2_df <- do.call("rbind", out2)
plot(out2_df[,1], train.x[,1])
plot(out2_df[,2], train.x[,2])
plot(out2_df[,3], train.x[,3])
plot(out2_df[,4], train.x[,4])
