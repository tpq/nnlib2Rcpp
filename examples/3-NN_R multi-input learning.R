##############################################################
### This example builds on the multi-task learning example to
### show how computed differentials can be passed between
### discrete models to enable multi-input learning. Unlike
### the previous example, this example makes use of NN_R6
### helper functions, including function piping and the
### $pass_forward() and $pass_backward() methods.
##############################################################

library(nnlib2Rcpp)
library(magrittr)

# scale data from 0 to 1 range
iris_s <- as.matrix( iris [ 1 : 4 ] )
c_min <- apply( iris_s, 2, FUN = "min" )
c_max <- apply( iris_s, 2, FUN = "max" )
c_rng <- c_max - c_min
iris_s <- sweep( iris_s, 2, FUN="-", c_min )
iris_s <- sweep( iris_s, 2, FUN="/", c_rng )

# 1-hot encode class labels
mat <- model.matrix(~.+0, iris[,5,drop=F])

# Encode features from iris into two latent spaces
input1 <-
	NN_R$new() %>%
	from_input(2) %>%
	connect_to(size = 4, activation = "sigmoid", output_layer = FALSE) %>%
	connect_to(size = 1, activation = "sigmoid", output_layer = FALSE) %>%
	ready()

input2 <-
	NN_R$new() %>%
	from_input(2) %>%
	connect_to(size = 4, activation = "sigmoid", output_layer = FALSE) %>%
	connect_to(size = 1, activation = "sigmoid", output_layer = FALSE) %>%
	ready()

# Then re-construct the input
decode_to_self <-
	NN_R$new() %>%
	from_input(2) %>%
	connect_to(size = 4, activation = "sigmoid", output_layer = FALSE) %>%
	connect_to(size = 4, activation = "sigmoid", output_layer = FALSE) %>%
	connect_to(size = 4, activation = "linear", output_layer = TRUE) %>%
	ready()

# Train the network using the pass_forward() and pass_backward() helpers
train.x <- iris_s
train.y <- mat
epochs <- 1000
for(e in 1:epochs){ # for E epochs
	if(e %% 10 == 0) print(e)
	for(r in sample(1:nrow(train.x))){
		# Forward pass
		latent1 <- input1$pass_forward(train.x[r,1:2])
		latent2 <- input2$pass_forward(train.x[r,3:4])
		res <- decode_to_self$pass_forward(c(latent1, latent2))

		# Backward pass
		delta <- decode_to_self$pass_backward(train.x[r,])
		input1$pass_backward(delta[1])
		input2$pass_backward(delta[2])
	}
}

z1 <- vector("list", nrow(train.x))
z2 <- vector("list", nrow(train.x))
final <- vector("list", nrow(train.x))
for(r in 1:nrow(train.x)){
	latent1 <- input1$pass_forward(train.x[r,1:2])
	latent2 <- input2$pass_forward(train.x[r,3:4])
	z1[[r]] <- latent1
	z2[[r]] <- latent2
	res <- decode_to_self$pass_forward(c(latent1, latent2))
	final[[r]] <- res
}

# View auto-encoder results
final_df <- do.call("rbind", final)
plot(final_df[,1], train.x[,1])
plot(final_df[,2], train.x[,2])
plot(final_df[,3], train.x[,3])
plot(final_df[,4], train.x[,4])

# View latent space
z1_df <- do.call("rbind", z1)
z2_df <- do.call("rbind", z2)
plot(z1_df)
plot(z2_df)
