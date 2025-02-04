//		----------------------------------------------------------
//		(c)2019  Vasilis.N.Nikolaidis          All rights reserved.
//		-----------------------------------------------------------
//		layer.h		 							Version 0.1
//		-----------------------------------------------------------
//		Definition - implementation of general neural layer
//		(layer) functionality.
//		-----------------------------------------------------------


#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "component.h"
#include "pe.h"
#include "nnlib2_vector.h"
#include "nnlib2_misc.h"

namespace nnlib2 {

/*-----------------------------------------------------------------------*/
/* Neural layer virtual (common methods,regardless of PE_TYPE)           */
/*-----------------------------------------------------------------------*/

class layer : public data_io_component, public error_flag_client                                                      // a component that can input and output data from/to its environment
{
protected:
	virtual bool move_all_pe_input_to_output() = 0;
public:
	virtual void reset() = 0;
	virtual bool setup(string name, int size) = 0;
	virtual void draw() = 0;
	virtual int size() = 0;
	int length() {return size();}
	virtual pe REF PE(int pe) = 0;                                                      // note: this is not PE_TYPE specific (pe)
	virtual void randomize_biases(DATA min_random_value, DATA max_random_value) = 0;
	virtual string item_description(int item) = 0;
	virtual void from_stream(std::istream REF s) = 0;                                   // read layer from stream
	virtual void to_stream(std::ostream REF s) = 0;                                     // write layer to stream
	virtual bool set_output(DATA * data, int dimension) = 0;
	virtual bool set_biases(DATA * data, int dimension) = 0;							// added for nnlib2Rcpp 0.1.10
	virtual bool set_bias_at(int index, DATA d) = 0;									// added for nnlib2Rcpp 0.1.10
	virtual bool get_biases(DATA * buffer, int dimension) = 0;							// added for nnlib2Rcpp 0.1.10
	virtual bool get_misc(DATA * buffer, int dimension) = 0;							// added for nnlib2Rcpp 0.1.11
	virtual bool set_misc(DATA * data, int dimension) = 0;
	virtual DATA get_bias_from(int index) = 0;											// added for nnlib2Rcpp 0.1.10
	virtual bool get_input(DATA * buffer, int dimension) = 0;						    // added for nnlib2Rcpp 0.2.0
};

/*-----------------------------------------------------------------------*/
/* Neural Layer	(template for pe or pe-derived objects)                  */
/*-----------------------------------------------------------------------*/

template <class PE_TYPE>
class Layer : public layer
{
protected:

	vector <PE_TYPE> pes;                                                       // Processing Elements in layer.

	bool move_all_pe_input_to_output();

public:

	Layer();
	Layer(string name, int size);
	Layer(string name, int size, bool PTR error_flag_to_use);
	~Layer();
	void reset();
	bool setup(string name, int size);
	bool setup(string name, int size, bool PTR error_flag_to_use);
	void draw();
	int size();
	pe REF PE(int pe);
	void randomize_biases (DATA min_random_value,DATA max_random_value);
	string item_description(int item);
	void from_stream (std::istream REF s);                                 // read layer from stream
	void to_stream (std::ostream REF s);                                   // write layer to stream

	bool input_data_from_vector(DATA * data, int dimension);               // overrides virtual method in data_receiver, sets values to pe inputs
	bool output_data_to_vector(DATA * buffer, int dimension);              // overrides virtual method in data_provider, gets values from pe outputs
	bool send_input_to(int index, DATA d);                                 // overrides virtual method in data_receiver, sets value to corresponding pe input sets this input to respective pe input (and also appends to pe's list of input values)
	DATA get_output_from (int index);                                      // overrides virtual method in data_provider, gets value from corresponding pe output

	bool get_misc(DATA * buffer, int dimension);						   // added for nnlib2Rcpp 0.1.11
	bool set_misc(DATA * data, int dimension);							   // set values in misc internal register variables in layer pes

	bool set_output(DATA * data, int dimension);						   // overwrites current output registers in layer pes, with the provided data values

	bool set_biases(DATA * data, int dimension);						   // overwrites current bias registers in layer pes, with the provided data values
	bool set_bias_at(int index, DATA d);								   // overwrites current bias register in pe, with the provided data values
	bool get_biases(DATA * buffer, int dimension);						   // added for nnlib2Rcpp 0.1.10
	DATA get_bias_from(int index);										   // added for nnlib2Rcpp 0.1.10
	bool get_input(DATA * buffer, int dimension);						   // added for nnlib2Rcpp 0.2.0

	void encode();                                                         // (virtual in component) may be overridden by derived classes with specific layer functiobality.
	void recall();                                                         // (virtual in component) may be overridden by derived classes with specific layer functiobality.

	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

};

//-------------------------------------------------------------------------
// Neural Layer implementation follows:

template <class PE_TYPE>
Layer<PE_TYPE>::Layer()
{
	m_type = cmpnt_layer;
	m_name = "uninitialized zero-sized unnamed layer";
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <class PE_TYPE>
Layer<PE_TYPE>::Layer(string name, int size)
{
	m_type = cmpnt_layer;
	setup(name, size);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <class PE_TYPE>
Layer<PE_TYPE>::Layer(string name, int size, bool PTR error_flag_to_use)
{
	m_type = cmpnt_layer;
	setup(name, size, error_flag_to_use);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <class PE_TYPE>
Layer<PE_TYPE>::~Layer() { reset(); }

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <class PE_TYPE>
void Layer<PE_TYPE>::reset()
{
	pes.set_error_flag(my_error_flag());
	pes.reset();
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <class PE_TYPE>
bool Layer<PE_TYPE>::setup(string name, int size)
{
	if (no_error())
	{
		reset();

		if (size < 1)
		{
			error(NN_INTEGR_ERR, "Invalid layer size (<1)");
			return false;
		}
		m_name = name;
		pes.setup(size);
	}
	return no_error();
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <class PE_TYPE>
bool Layer<PE_TYPE>::setup(string name, int size, bool PTR error_flag_to_use)
{
	set_error_flag(error_flag_to_use);
	return setup(name,size);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <class PE_TYPE>
void Layer<PE_TYPE>::draw()
{
	if (no_error()) {}
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <class PE_TYPE>
int Layer<PE_TYPE>::size()
{
	return pes.number_of_items();
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <class PE_TYPE>
pe REF Layer<PE_TYPE>::PE(int pe)
{
	return pes[pe];
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <class PE_TYPE>
void Layer<PE_TYPE>::randomize_biases(DATA min_random_value, DATA max_random_value)
{
	for (int i = 0; i < size(); i++)
		pes[i].bias = random(min_random_value, max_random_value);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <class PE_TYPE>
string Layer<PE_TYPE>::item_description(int item)
{
	return pes[item].description();
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// input it :

template <class PE_TYPE>
void Layer<PE_TYPE>::from_stream(std::istream REF s)
{
	if (no_error())
	{
		component::from_stream(s);
		pes.from_stream(s);								// changed for VC7 port,was	s >> pes;
	}
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// output it :

template <class PE_TYPE>
void Layer<PE_TYPE>::to_stream(std::ostream REF s)
{
	if (no_error())
	{
		component::to_stream(s);
		pes.to_stream(s);								// changed for VC7 port,was	s << pes;
	}
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// overrides virtual method in data_receiver, sets values to pe inputs
// (sets this input to respective pe input and to received_values,
// clearing all previous ones)

template <class PE_TYPE>
bool Layer<PE_TYPE>::input_data_from_vector(DATA* data, int dimension)
{
	if (NOT no_error()) return false;
	if (data == NULL) return false;
	if (dimension NEQL size())
	{ warning ("Incompatible vector dimension (number of PEs vs vector length)");
		return false; }
	for (int i = 0; i < dimension; i++)
	{
		pes[i].input = data[i];                    // sets this input to respective pe input for direct processing
		pes[i].reset_received_values();
		pes[i].receive_input_value(data[i]);       // make this the only received_value (for optional processing by pe::input_function)
	}
	return true;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// overrides virtual method in data_provider, gets values from pe outputs
// (gets the output value from respective pes and copies it to buffer)

template <class PE_TYPE>
bool Layer<PE_TYPE>::output_data_to_vector(DATA* buffer, int dimension)
{
	if (NOT no_error()) return false;
	if (buffer == NULL) return false;
	if (dimension NEQL size()) { warning ("Incompatible output vector dimension (number of PEs vs vector length)"); return false; }
	for (int i = 0; i < dimension; i++) buffer[i] = pes[i].output;
	return true;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// overrides virtual method in data_receiver, sets value to corresponding pe input

template <class PE_TYPE>
bool Layer<PE_TYPE>::send_input_to(int index, DATA d)
{
	if (NOT no_error()) return false;
	if (index < 0) return false;
	if (index >= size()) { error(NN_INTEGR_ERR, "Cannot access PE at this index position"); return false; }
	pes[index].input = d;
	return true;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// overrides virtual method in data_provider, gets value from corresponding pe output

template <class PE_TYPE>
DATA Layer<PE_TYPE>::get_output_from(int index)
{
	if (NOT no_error()) return false;
	if (index < 0) return false;
	if (index >= size()) { error(NN_INTEGR_ERR, "Cannot access PE at this index position"); return false; }
	return(pes[index].output);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// get misc variable values from pes

template <class PE_TYPE>
bool Layer<PE_TYPE>::get_misc(DATA * buffer, int dimension)
{
	if (NOT no_error()) return false;
	if (buffer == NULL) return false;
	if (dimension NEQL size())
	{ warning ("Incompatible vector dimension (number of PEs vs vector length)");
		return false; }
	for (int i = 0; i < dimension; i++)
		buffer[i] = pes[i].misc;                    // gets respective pe misc
	return true;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// set misc variable in pes

template <class PE_TYPE>
bool Layer<PE_TYPE>::set_misc(DATA * data, int dimension)
{
	if (NOT no_error()) return false;
	if (data == NULL) return false;
	if (dimension NEQL size())
	{ warning ("Incompatible vector dimension (number of PEs vs vector length)");
		return false; }
	for (int i = 0; i < dimension; i++)
		pes[i].misc = data[i];                    // sets data to respective pe misc
	return true;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// set output variable in pes (overwriting current output values)

template <class PE_TYPE>
bool Layer<PE_TYPE>::set_output(DATA * data, int dimension)
{
	if (NOT no_error()) return false;
	if (data == NULL) return false;
	if (dimension NEQL size())
	{ warning ("Incompatible vector dimension (length)");
		return false; }
	for (int i = 0; i < dimension; i++)
		pes[i].output = data[i];                    // sets data to respective pe output
	return true;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// set bias variable in pes (overwriting current bias values)
// added for nnlib2Rcpp 0.1.10 (see Github issue https://github.com/VNNikolaidis/nnlib2Rcpp/issues/13)

template <class PE_TYPE>
bool Layer<PE_TYPE>::set_biases(DATA * data, int dimension)
{
	if (NOT no_error()) return false;
	if (data == NULL) return false;
	if (dimension NEQL size())
	{ warning ("Incompatible vector dimension (length)");
		return false; }
	for (int i = 0; i < dimension; i++)
		pes[i].bias = data[i];                    // sets data to respective pe bias
	return true;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// set bias variable in specified  pe (overwriting current bias value)
// added for nnlib2Rcpp 0.1.10 (see Github issue https://github.com/VNNikolaidis/nnlib2Rcpp/issues/13)

template <class PE_TYPE>
bool Layer<PE_TYPE>::set_bias_at(int index, DATA d)
{
	if (NOT no_error()) return false;
	if (index < 0)
         { warning("No PE at specified index (numbering starts from 0)"); return false; }
	if (index >= size())
		{ warning("No PE at specified index (numbering starts from 0)"); return false; }
	pes[index].bias = d;
	return true;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// get bias values
// added for nnlib2Rcpp 0.1.10 (see Github issue https://github.com/VNNikolaidis/nnlib2Rcpp/issues/13)

template <class PE_TYPE>
bool Layer<PE_TYPE>::get_biases(DATA * buffer, int dimension)
{
	if (NOT no_error()) return false;
	if (buffer == NULL) return false;
	if (dimension NEQL size()) { warning ("Incompatible output vector dimension (number of PEs vs vector length)"); return false; }
	for (int i = 0; i < dimension; i++) buffer[i] = pes[i].bias;
	return true;
}



//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// get (estimate of) input values
//
// Important!!! see implementation of preview_current_input as this is an ESTIMATE and may be affected by PE specific implementation!

template <class PE_TYPE>
bool Layer<PE_TYPE>::get_input(DATA * buffer, int dimension)
	{
		if (NOT no_error()) return false;
		if (buffer == NULL) return false;
		if (dimension NEQL size()) { warning ("Incompatible output vector dimension (number of PEs vs vector length)"); return false; }
		for (int i = 0; i < dimension; i++) buffer[i] = pes[i].preview_current_input();
		return true;
	}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// get bias values
// added for nnlib2Rcpp 0.1.10 (see Github issue https://github.com/VNNikolaidis/nnlib2Rcpp/issues/13)

template <class PE_TYPE>
DATA Layer<PE_TYPE>::get_bias_from(int index)
{

	if (NOT no_error()) return 0;
	if (index < 0)
	{ warning("No PE at specified index (numbering starts from 0)"); return 0; }
	if (index >= size())
	{ warning("No PE at specified index (numbering starts from 0)"); return 0; }
	return pes[index].bias;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// copies inputs to outputs, also zeros inputs

template <class PE_TYPE>
bool Layer<PE_TYPE>::move_all_pe_input_to_output()
{
	if (no_error())
		for (int i = 0; i < size(); i++) pes[i].move_input_to_output();
	return (no_error());
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// should be overridden by derived classes.

template <class PE_TYPE>
void Layer<PE_TYPE>::encode()
{
	if (no_error())
		for (int i = 0; i < size(); i++) pes[i].encode();
}

template <class PE_TYPE>
void Layer<PE_TYPE>::recall()
{
	if (no_error())
		for (int i = 0; i < size(); i++) pes[i].recall();
}

//-------------------------------------------------------------------------
// Various layer-related useful definitions:

//-------------------------------------------------------------------------
// For explicit instantiation of layer template per pe type (not needed
// as implementation is in this header) use code similar to:


// template class layer<pe>;   // (instantiate a layer of generic pes)

//-------------------------------------------------------------------------
// layer of generic "dumb" pes where most processing will be done in layer code:

typedef Layer<pe> pe_layer;

//-------------------------------------------------------------------------
// layer of generic "dumb" pes that only passes pe input variable to output:
// note: this uses the pe internal input _variable_ (does not process inputs)

class pass_through_layer : public pe_layer
{
public:
	pass_through_layer(string name, int size):pe_layer(name,size){}
	void encode() { move_all_pe_input_to_output(); }
	void recall() { move_all_pe_input_to_output(); }
};

//-------------------------------------------------------------------------

}   // end of namespace nnlib2

#endif // NN_LAYER_H
