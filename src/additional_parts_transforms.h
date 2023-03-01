#ifndef NNLIB2_ADDITIONAL_PARTS_TRANSFORMS_H
#define NNLIB2_ADDITIONAL_PARTS_TRANSFORMS_H

#include "nn.h"
using namespace nnlib2;

//--------------------------------------------------------------------------------------------
// linear only (no transform)

class bp_comput_layer_v2: public bp::bp_comput_layer
{
public:

	bool use_bias = true;
	void set_use_bias(bool new_value)
	{
		use_bias = new_value;
	}

	bool output_layer = false;
	void set_output_layer(bool new_value)
	{
		output_layer = new_value;
	}

	std::string activation = "linear";
	void set_activation(std::string new_value)
	{
		activation = new_value;
	}

	void recall()
	{
		if(no_error())
		{
			for(int i=0;i<size();i++)
			{
				pe REF p = pes[i];
				DATA x = p.input; // input is already summated
				if(use_bias) x = x + p.bias; // add bias is optional
				if(activation == "linear"){ // different recall formula for each activation
					p.output = (DATA)1*x;
				}else if(activation == "relu"){
					if(x > 0){
						p.output = (DATA)1*x;
					}else{
						p.output = (DATA)1*0;
					}
				}else if(activation == "sigmoid"){
					p.output = (DATA)1 / (1 + exp(-x));
				}else if(activation == "tanh"){
					p.output = (DATA)1*tanh(x);
				}else{
					error(NN_ARITHM_ERR, "Activation not recognised.");
				}
				p.input = 0; // reset input
			}
		}
	}

	void encode()
	{
		if(no_error())
		{
			for(int i=0;i<size();i++)
			{
				pe REF p = pes[i];
				DATA current = p.output; // here is the last output produced

				float h_prime_of_current; // compute derivative of current value
				if(activation == "linear"){ // different derivative for each activation
					h_prime_of_current = 1;
				}else if(activation == "relu"){
					if(current > 0){
						h_prime_of_current = 1;
					}else{
						h_prime_of_current = 0;
					}
				}else if(activation == "sigmoid"){
					h_prime_of_current = current * (1 - current);
				}else if(activation == "tanh"){
					h_prime_of_current = 1 - pow(tanh(current), 2);
				}else{
					error(NN_ARITHM_ERR, "Activation not recognised.");
				}

				DATA pass_backward;
				if(output_layer){
					DATA desired = p.input;
					pass_backward = h_prime_of_current * ( desired - current ); // if output layer, use 'desired - current'
				}else{
					DATA d = p.input;
					pass_backward = h_prime_of_current * ( d ); // if not output layer, pass along delta from next layer
				}

				p.misc = pass_backward;	// store relative error in 'misc' -- this is passed to connection set
				p.input = 0; // reset input (again)
				p.bias += m_learning_rate * pass_backward; // adjust bias (SIMPSON 5-167)
			}
		}
	}
};

#endif // NNLIB2_ADDITIONAL_PARTS_TRANSFORMS_H
