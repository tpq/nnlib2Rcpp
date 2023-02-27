library(nnlib2Rcpp)

test_that("linear activation with biases works", {

	a <- NN_R$new()
	a$add_layer(name = "generic", size = 4)
	a$add_connection_set( "BP" )
	a$add_compute_layer(size = 3, activation = "linear", use_bias = TRUE)
	a$create_connections_in_sets(-1, 1)

	a$input_at(1, c(1, 1, 1, 1))
	a$recall_all()
	X <- a$get_input_at(1)
	W <- a$get_weights_at(2)
	B <- a$get_biases_at(3)
	Y <- a$get_output_from(3)

	expect_equal(
		as.numeric(X %*% matrix(W, 4, 3) + B),
		Y
	)
})

test_that("linear activation without biases works", {

	a <- NN_R$new()
	a$add_layer(name = "generic", size = 4)
	a$add_connection_set( "BP" )
	a$add_compute_layer(size = 3, activation = "linear", use_bias = FALSE)
	a$create_connections_in_sets(-1, 1)

	a$input_at(1, c(1, 1, 1, 1))
	a$recall_all()
	X <- a$get_input_at(1)
	W <- a$get_weights_at(2)
	B <- a$get_biases_at(3)
	Y <- a$get_output_from(3)

	expect_equal(
		as.numeric(X %*% matrix(W, 4, 3) + 0),
		Y
	)
})

test_that("non-implemented activations throw error", {

	a <- NN_R$new()
	a$add_layer(name = "generic", size = 4)
	a$add_connection_set( "BP" )
	a$add_compute_layer(size = 3, activation = "error", use_bias = TRUE)
	a$create_connections_in_sets(-1, 1)
	a$input_at(1, c(1, 1, 1, 1))

	expect_error(
		a$recall_all()
	)
})
