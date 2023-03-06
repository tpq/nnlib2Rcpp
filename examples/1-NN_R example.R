##############################################################
### This example introduces the NN_R class object to show how
### it can be used similar to the C++ class.
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

# use NN_R class
a <- NN_R$new()
a$add_layer(name = "generic", size = 4)
a$add_connection_set( "BP" )
a$add_compute_layer(size = 3, activation = "linear", output_layer = TRUE)
a$create_connections_in_sets (-1, 1)
a$outline()

train_nnet <- function(model, train.x, train.y, epochs){

	final_layer <- length(model$sizes())
	for(e in 1:epochs) # for E epochs

		for(r in sample(1:nrow(train.x))) # for each data case
		{
			# present data at 1st layer
			model$input_at( 1, train.x[r,,drop=F] )
			# recall (fwd direction, entire topology)
			model$recall_all( TRUE )
			# present data at last layer
			model$input_at( final_layer, train.y[r,,drop=F] )
			# encode, adjusting weights (bwd-direction in topology)
			model$encode_all ( FALSE )
		}

	return(model)
}

predict_nnet <- function(model, test.x){

	final_layer <- length(model$sizes())
	results <- lapply(1:nrow(test.x), function(r){

		# present data at 1st layer
		model$input_at( 1, test.x[r,,drop=T] )
		# recall (fwd direction, entire topology)
		model$recall_all( TRUE )
		model$get_output_at( final_layer )
	})

	do.call('rbind', results)
}

train_nnet(a, iris_s, mat, 100)
res <- predict_nnet(a, iris_s)
plot(res[,1])
plot(res[,2])
plot(res[,3])
