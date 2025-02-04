#ifndef NNLIB2_ADDITIONAL_PARTS_OTHER_H
#define NNLIB2_ADDITIONAL_PARTS_OTHER_H

#include "nn.h"
using namespace nnlib2;

//--------------------------------------------------------------------------------------------
// example for manual.pdf (vignette)

class MEX_connection: public connection
{
public:

	// model-specific behavior during mapping stage:

	void recall()
	{
		destin_pe().receive_input_value(pow( source_pe().output - weight() , 2) );
	}

	// model-specific behavior during training stage:
	// in this example, only the current  connection weight (i.e. weight())
	// and incoming value from the source node (i.e. source_pe().output) are
	// used in a series of calculations that eventually change the
	// connection weight (see last line).

	void encode()
	{
		double x  = source_pe().output - weight();
		double s  = .3;
		double m  = weight();
		double pi = 2*acos(0.0);
		double f  = 1/(s*sqrt(2*pi)) * exp(-0.5*pow((x-m)/s,2));
		double d = (f * x) / 2;
		weight() = weight() + d;
	}

};

typedef Connection_Set < MEX_connection > MEX_connection_set;

//--------------------------------------------------------------------------------------------
// example for manual.pdf (vignette)

class MEX_pe : public pe
{
public:

	void recall()
	{
		pe::recall();
		output = sqrt(output);
	}
};

typedef Layer < MEX_pe > MEX_layer;

//--------------------------------------------------------------------------------------------
// example (for RBloggers blog post): Perceptron components
// see: https://www.r-bloggers.com/2020/07/creating-custom-neural-networks-with-nnlib2rcpp/

// Perceptron nodes:
class perceptron_pe : public pe
{
public:
	DATA threshold_function(DATA value)
	{
		if(value>0) return 1;
		return 0;
	}
};

// Percepton layer:
typedef Layer < perceptron_pe > perceptron_layer;

// Perceptron connections:
class perceptron_connection: public connection
{
public:

	// multiply incoming (from source node) value by weight and send it to destination node.
	void recall()
	{
		destin_pe().receive_input_value( weight() * source_pe().output );
	}

	// for simplicity, learning rate is fixed to 0.3 and input contains desired output:
	void encode()
	{
		weight() = weight() + 0.3 * (destin_pe().input - destin_pe().output) * source_pe().output;
	}
};

// Perceptron group of connections
typedef Connection_Set< perceptron_connection > perceptron_connection_set;

//--------------------------------------------------------------------------------------------
// example: a (useless) pe that just adds 10 to the sum of its inputs when recalling data

class JustAdd10_pe : public pe
{
public:
	void recall() {	pe::recall(); output = output + 10; }
};

typedef Layer < JustAdd10_pe > JustAdd10_layer;

//--------------------------------------------------------------------------------------------

#endif // NNLIB2_ADDITIONAL_PARTS_OTHER_H
