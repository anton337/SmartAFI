#ifndef AFIfunctor_h
#define AFIfunctor_h

#include "fft.h"

#include "transpose.h"

#include "semblance.h"

#include "shear.h"

#include "display_update.h"

#include "array_verify.h"

#include "compute_device.h"

class AFIfunctor
{
	DisplayUpdate * global_display_update;
	DisplayUpdate *   tile_display_update;
public:
	AFIfunctor	( DisplayUpdate * _global_display_update
				      , DisplayUpdate *   _tile_display_update
				      )
	{
		global_display_update = _global_display_update;
		  tile_display_update =   _tile_display_update;
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

	  std::string input_name = "input";            // note to self: could potentially delete this after getting transpose, if memory is an issue
	  std::string transpose_name = "transpose";    // can also get rid of this guy, once the semblance volumes have been calculated
	  std::string numerator_name = "numerator";
	  std::string denominator_name = "denominator";
	  std::string numerator_name_freq = "numerator_freq";
	  std::string denominator_name_freq = "denominator_freq";
	  std::string rotation_kernel_name = "rotation_kernel";
	  std::string rotated_numerator_name = "rotated_numerator";
	  std::string rotated_denominator_name = "rotated_denominator";
	  std::string rotated_numerator_name_freq = "rotated_numerator_freq";
	  std::string rotated_denominator_name_freq = "rotated_denominator_freq";
	  std::string rotated_numerator_name_time = "rotated_numerator_time";
	  std::string rotated_denominator_name_time = "rotated_denominator_time";
	  std::string sheared_numerator_name_freq = "sheared_numerator_freq";
	  std::string sheared_denominator_name_freq = "sheared_denominator_freq"; // this also serves as the semblance output, but it is repopulated with fresh data with each shear iteration, so it's ok
	  std::string sheared_numerator_name_time = "sheared_numerator_time";
	  std::string sheared_denominator_name_time = "sheared_denominator_time";
	  std::string fault_likelihood_name = "fault_likelihood";
	  std::string optimal_fault_likelihood_name = "optimal_fault_likelihood";
	  std::string output_fault_likelihood_name = "output_fault_likelihood";

    {
      std::cout << "p1" << std::endl;
	  	d->create(input_name, nx, ny, nz, in, false, true); // allocate input array (time domain) {X,Y,Z}
      std::cout << "p2" << std::endl;

	  	d->create(transpose_name, nz, ny, nx); // create transpose array (time domain) {Z,Y,X}
	  	d->compute_transpose(nx, ny, nz, input_name, transpose_name);
      std::cout << "p3" << std::endl;

	  	d->init_fft(nz, ny, nx);
      std::cout << "p4" << std::endl;

	  	d->initialize_semblance(nz,ny,nx,transpose_name,numerator_name,denominator_name,numerator_name_freq,denominator_name_freq);
      std::cout << "p5" << std::endl;

	  	d->create(optimal_fault_likelihood_name, nz, ny, nx);
	  	d->create(output_fault_likelihood_name, nx, ny, nz);
      std::cout << "p6" << std::endl;

	  	d->create(rotated_numerator_name_freq, nz, ny, nx, NULL, true);
	  	d->create(rotated_denominator_name_freq, nz, ny, nx, NULL, true);
      std::cout << "p7" << std::endl;

	  	d->create(rotated_numerator_name_time, nz, ny, nx, NULL, true);
	  	d->create(rotated_denominator_name_time, nz, ny, nx, NULL, true);
      std::cout << "p8" << std::endl;

	  	d->create(sheared_numerator_name_freq, nz, ny, nx, NULL, true);
	  	d->create(sheared_denominator_name_freq, nz, ny, nx, NULL, true);
      std::cout << "p9" << std::endl;

	  	d->create(sheared_numerator_name_time, nz, ny, nx, NULL, true);
	  	d->create(sheared_denominator_name_time, nz, ny, nx, NULL, true);
      std::cout << "p10" << std::endl;

	  	d->init_shear(nz, ny, nx, fault_likelihood_name, sheared_numerator_name_freq, sheared_denominator_name_freq);
      std::cout << "p11" << std::endl;

	  	int theta_ind = 0;
	  	for (float theta = -M_PI; theta <= 0; theta += M_PI / 64, theta_ind++)
	  	{
	  		std::stringstream ss_rotation_kernel_name_freq;
	  		ss_rotation_kernel_name_freq << rotation_kernel_name << "-" << theta_ind << "-freq";
	  		if (d->create(ss_rotation_kernel_name_freq.str(), 1, ny, nx, NULL, true)) // create frequency domain array for kernel
	  		{
	  			std::stringstream ss_rotation_kernel_name_time;
	  			ss_rotation_kernel_name_time << rotation_kernel_name << "-" << theta_ind << "-time";
	  			d->initialize_rotation_kernel(nx, ny, theta, ss_rotation_kernel_name_time.str(), ss_rotation_kernel_name_freq.str());
	  		}
	  		
	  		d->compute_convolution_rotation_kernel(nz, ny, nx, ss_rotation_kernel_name_freq.str(), numerator_name_freq, rotated_numerator_name_freq);
	  		
	  		d->compute_convolution_rotation_kernel(nz, ny, nx, ss_rotation_kernel_name_freq.str(), denominator_name_freq, rotated_denominator_name_freq);

	  		float shear_extend = 0.1f;
	  		for (float shear = -shear_extend; shear <= shear_extend; shear += shear_extend/32.0f)
	  	//	float shear = 0.0f;
	  		{

	  			float shear_y = shear*cos(theta);
	  			float shear_x = -shear*sin(theta);

	  			d->compute_fault_likelihood(
	  				nz
	  				, ny
	  				, nx
	  				, shear_y
	  				, shear_x
	  				, rotated_numerator_name_freq
	  				, rotated_denominator_name_freq
	  				, sheared_numerator_name_freq
	  				, sheared_denominator_name_freq/*this guys is recycled internally, which is fine, since it is repopulated with fresh data with each shear iteration*/
	  				, fault_likelihood_name/*time domain {Z,Y,X}*/
	  				, optimal_fault_likelihood_name
	  				//, optimal_theta_name
	  				//, optimal_phi_name
	  				);

	  		}

	  	}

	  	d->destroy_fft();
      std::cout << "p100" << std::endl;

	  	d->compute_transpose(nz, ny, nx, optimal_fault_likelihood_name, output_fault_likelihood_name);
      std::cout << "p101" << std::endl;

	  	d->destroy(input_name);

	  }
	
  }

};

#endif
