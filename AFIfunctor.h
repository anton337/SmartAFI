#ifndef AFIfunctor_h
#define AFIfunctor_h

#include "fft.h"

#include "transpose.h"

#include "semblance.h"

#include "shear.h"

#include "display_update.h"

#include "array_verify.h"

#include "compute_device.h"

#include "fault_sorter.h"

class AFIfunctor
{
	DisplayUpdate * global_display_update;
	DisplayUpdate *   tile_display_update;
	float * tile, *tile1, *tile2;
public:
	AFIfunctor	( DisplayUpdate * _global_display_update
				, DisplayUpdate *   _tile_display_update
				)
	{
		global_display_update = _global_display_update;
		  tile_display_update =   _tile_display_update;
		  tile = NULL;
                  tile1 = NULL;
                  tile2 = NULL;
	}
  
	void operator()(
		std::size_t nx
		, std::size_t ny
		, std::size_t nz
		, std::size_t pad
		, float * in // {X,Y,Z}
		, ComputeDevice * d
		)
	{

		if (tile == NULL)
		{
			tile = new float[nx*ny*nz];
		}
		if (tile1 == NULL)
		{
			tile1 = new float[nx*ny*nz];
		}
		if (tile2 == NULL)
		{
			tile2 = new float[nx*ny*nz];
		}

	  float scalar_rotation = 64;
    	  float scalar_shear = 8;
    	  float thin_threshold = 0.0f;

	  Token input_token ("input", GPU, nx, ny, nz);            // note to self: could potentially delete this after getting transpose, if memory is an issue
	  Token transpose_token("transpose", GPU, nz, ny, nx);    // can also get rid of this guy, once the semblance volumes have been calculated
	  Token numerator_token("numerator", GPU, nz, ny, nx);
	  Token denominator_token("denominator", GPU, nz, ny, nx);
	  Token numerator_token_freq("numerator_freq", FREQ_GPU, nz, ny, nx);
	  Token denominator_token_freq("denominator_freq", FREQ_GPU, nz, ny, nx);
	  Token rotation_kernel_token("rotation_kernel", GPU, nz, ny, nx);
	  Token rotated_numerator_token("rotated_numerator", GPU, nz, ny, nx);
	  Token rotated_denominator_token("rotated_denominator", GPU, nz, ny, nx);
	  Token rotated_numerator_token_freq("rotated_numerator_freq", FREQ_GPU, nz, ny, nx);
	  Token rotated_denominator_token_freq("rotated_denominator_freq", FREQ_GPU, nz, ny, nx);
	  Token sheared_numerator_token_freq("sheared_numerator_freq", FREQ_GPU, nz, ny, nx);
	  Token sheared_denominator_token_freq("sheared_denominator_freq", FREQ_GPU, nz, ny, nx); // this also serves as the semblance output, but it is repopulated with fresh data with each shear iteration, so it's ok
	  Token fault_likelihood_token("fault_likelihood", GPU, nz, ny, nx);
	  Token optimal_fault_likelihood_token("optimal_fault_likelihood", GPU, nz, ny, nx);
	  Token optimal_theta_token("optimal_theta", GPU, nz, ny, nx);
	  Token optimal_thin_token("optimal_thin", GPU, nz, ny, nx);
	  Token output_fault_likelihood_token("output_fault_likelihood", GPU, nx, ny, nz);
	  Token output_theta_token("output_theta", GPU, nx, ny, nz);
	  Token output_thin_token("output_thin", GPU, nx, ny, nz);

    {
	  	d->create(input_token, nx, ny, nz, in, false, true); // allocate input array (time domain) {X,Y,Z}
	  	d->create(transpose_token, nz, ny, nx, NULL, false); // create transpose array (time domain) {Z,Y,X}
	  	d->compute_transpose(nx, ny, nz, input_token, transpose_token);

		//if(d->get_index()==0)d->get_output(transpose_token, tile1);
		if(d->get_index()==0)d->get_output(transpose_token, tile1);
		//usleep(1000);
		//if(d->get_index()==0)tile_display_update->update("input", nz, ny, nx, tile1);
		if(d->get_index()==0)tile_display_update->update("input", nz, ny, nx, tile1);

	  	d->init_fft(nz, ny, nx);
	  	d->initialize_semblance(nz,ny,nx,transpose_token,numerator_token,denominator_token,numerator_token_freq,denominator_token_freq);
	  	d->create(optimal_fault_likelihood_token, nz, ny, nx);
	  	d->create(optimal_theta_token, nz, ny, nx);
	  	d->create(optimal_thin_token, nz, ny, nx);
	  	d->create(output_fault_likelihood_token, nx, ny, nz);
	  	d->create(output_theta_token, nx, ny, nz);
	  	d->create(output_thin_token, nx, ny, nz);
	  	d->create(rotated_numerator_token_freq, nz, ny, nx, NULL, true);
	  	d->create(rotated_denominator_token_freq, nz, ny, nx, NULL, true);
	  	d->create(sheared_numerator_token_freq, nz, ny, nx, NULL, true);
	  	d->create(sheared_denominator_token_freq, nz, ny, nx, NULL, true);
	  	d->init_shear(nz, ny, nx, fault_likelihood_token, sheared_numerator_token_freq, sheared_denominator_token_freq);
	  	int theta_ind = 0;
	  	for (float theta = -M_PI; theta <= 0; theta += M_PI / scalar_rotation, theta_ind++)
	  	{
	  		std::stringstream ss_rotation_kernel_token_freq;
	  		ss_rotation_kernel_token_freq << rotation_kernel_token.name << "-" << theta_ind << "-freq";
	  		if (d->create(
					Token(ss_rotation_kernel_token_freq.str(), FREQ_GPU, 1, ny, nx)
					, 1
					, ny
					, nx
					, NULL
					, true
					)
				) // create frequency domain array for kernel
	  		{
	  			std::stringstream ss_rotation_kernel_token_time;
	  			ss_rotation_kernel_token_time << rotation_kernel_token.name << "-" << theta_ind << "-time";
	  			d->initialize_rotation_kernel(
					nx
					, ny
					, theta
					, Token(ss_rotation_kernel_token_time.str(), GPU, 1, ny, nx)
					, Token(ss_rotation_kernel_token_freq.str(), FREQ_GPU, 1, ny, nx)
					);
	  		}
	  		
	  		d->compute_convolution_rotation_kernel(
				nz
				, ny
				, nx
				, Token(ss_rotation_kernel_token_freq.str(), FREQ_GPU, 1, ny, nx)
				, Token(numerator_token_freq.name, FREQ_GPU, nz, ny, nx)
				, Token(rotated_numerator_token_freq.name, FREQ_GPU, nz, ny, nx)
				);
	  		
	  		d->compute_convolution_rotation_kernel(
				nz
				, ny
				, nx
				, Token(ss_rotation_kernel_token_freq.str(), FREQ_GPU, 1, ny, nx)
				, Token(denominator_token_freq.name, FREQ_GPU, nz, ny, nx)
				, Token(rotated_denominator_token_freq.name, FREQ_GPU, nz, ny, nx)
				);

	  		float shear_extend = 0.2f;
	  		//for (float shear = -shear_extend; shear <= shear_extend; shear += shear_extend/scalar_shear)
	  		float shear = 0.0f;
	  		{

	  			float shear_y = shear*cos(theta);
	  			float shear_x = -shear*sin(theta);

	  			d->compute_fault_likelihood(
	  				nz
	  				, ny
	  				, nx
	  				, shear_y
	  				, shear_x
            , theta
	  				, rotated_numerator_token_freq
	  				, rotated_denominator_token_freq
	  				, sheared_numerator_token_freq
	  				, sheared_denominator_token_freq/*this guys is recycled internally, which is fine, since it is repopulated with fresh data with each shear iteration*/
	  				, fault_likelihood_token/*time domain {Z,Y,X}*/
	  				, optimal_fault_likelihood_token
	  				, optimal_theta_token
	  				//, optimal_phi_token
	  				);

	  		}

		    //if(d->get_index()==0)d->get_output(fault_likelihood_token, tile1);
	            //if(d->get_index()==0)tile_display_update->update1("fault likelihood", nz, ny, nx, tile1);

		    if(d->get_index()==0)d->get_output(optimal_fault_likelihood_token, tile);
	            if(d->get_index()==0)tile_display_update->update1("update fault likelihood", nz, ny, nx, tile);

	  	}


	  	d->destroy_fft();

      d->compute_thin(nz, ny, nx, thin_threshold, optimal_fault_likelihood_token, optimal_theta_token, optimal_thin_token);

	  	d->compute_transpose(nz, ny, nx, optimal_fault_likelihood_token, output_fault_likelihood_token);

	  	d->compute_transpose(nz, ny, nx, optimal_theta_token, output_theta_token);

	  	d->compute_transpose(nz, ny, nx, optimal_thin_token, output_thin_token);

		if(d->get_index()==0)d->get_output(optimal_thin_token, tile2);

      		//FaultSorter faultSorter;
      		//faultSorter(nz,ny,nx,tile2);

		if(d->get_index()==0)tile_display_update->update2("optimal thin", nz, ny, nx, tile2);

		//-->if(d->get_index()==0)d->get_output(output_fault_likelihood_token, in);
		//if(d->get_index()==0)d->get_output(output_thin_token, in);

	  	d->destroy(input_token);
	  	d->destroy(optimal_fault_likelihood_token);

		//d->list_status();

	  }
	
  }

};

#endif
