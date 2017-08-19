#ifndef compute_device_h
#define compute_device_h

#include "message_assert.h"

#include "data_array.h"

#include "cuda_stuff.h"

#include "token.h"

class ComputeDevice
{
	std::size_t index;
	std::string name;
	typedef std::map<Token, DataArray*, TokenComparator> map_type;
	map_type data_array;
public:
	ComputeDevice(std::string _name,std::size_t _index)
		: index(_index)
		, name(_name)
	{

	}
	std::size_t get_index()
	{
		return index;
	}
	void* get(Token token)
	{
		if (data_array.find(token) == data_array.end())
		{
			std::cout << "can't find array:" << token.name << std::endl;
			char ch;
			std::cin >> ch;
			exit(1);
		}
		return data_array[token]->get();
	}
	void set(Token token,float * dat)
	{
		if (data_array.find(token) == data_array.end())
		{
			std::cout << "can't find array:" << token.name << std::endl;
			char ch;
			std::cin >> ch;
			exit(1);
		}
		data_array[token]->set(dat);
	}
	virtual void get_output(Token token,void* ptr)
	{
		if (data_array.find(token) == data_array.end())
		{
			std::cout << "can't find array:" << token.name << std::endl;
			char ch;
			std::cin >> ch;
			exit(1);
		}
		data_array[token]->get_output(ptr);
	}
	virtual bool create(Token token, int nx, int ny, int nz, float * dat = NULL, bool freq = false, bool keep_data = false) = 0;
	bool put(Token token, DataArray * dat)
	{
		if (data_array.find(token) == data_array.end())
		{
			destroy(token);
			data_array[token] = dat;
			return true;
		}
		else
		{
			if (dat->get_size() > data_array[token]->get_size()
				|| dat->get_type() != data_array[token]->get_type())
			{
				destroy(token);
				data_array[token] = dat;
				return true;
			}
		}
		return false;
	}
	void destroy(Token token)
	{
		map_type::iterator it = data_array.find(token);
		if (it == data_array.end())
		{
			return;
		}
		it->second->destroy();
		data_array.erase(it);
	}
	void remove(Token token)
	{
		destroy(token);
	}
	void list_status()
	{
		std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
		std::cout << "%%   " << name << std::endl;
		std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
		map_type::iterator it = data_array.begin();
		while (it != data_array.end())
		{
			std::cout << "%%\t" << it->second->get_name() << '\t';
			std::cout << it->second->get_pretty_type() << '\t';
			std::cout << it->second->get_size() << '\t';
			std::cout << it->second->get_validation() << '\t';
			std::cout << std::endl;
			it++;
		}
		std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
	}
	void destroy()
	{
		map_type::iterator it = data_array.begin();
		while (it != data_array.end())
		{
			it->second->destroy();
			it++;
		}
		data_array.clear();
	}
	virtual void fft(int nz, int ny, int nx, Token in_token, Token out_token) = 0;
	void initialize_rotation_kernel(
		int nx
		, int ny
		, float theta
		, Token rotation_kernel_token_time
		, Token rotation_kernel_token_freq
		)
	{
		MESSAGE_ASSERT(
			rotation_kernel_token_time.type == GPU
			,  "argument type mismatch"
			);
		MESSAGE_ASSERT(
			rotation_kernel_token_freq.type == FREQ_GPU
			, "argument type mismatch"
			);
		float * kernel_data = new float[nx*ny];
		memset(kernel_data, 0, nx*ny);
		float s = 2.0;
		float sigma_x_2 = 0.005f / (s*s);
		float sigma_y_2 = 0.000002f*10.0f;
		float a, b, c;
		float cos_theta = cos(theta);
		float cos_theta_2 = cos_theta*cos_theta;
		float sin_theta = sin(theta);
		float sin_theta_2 = sin_theta*sin_theta;
		float sin_2_theta = sin(2 * theta);
		float Z;
		float dx, dy;
		for (int y = 0, k = 0; y < ny; y++)
		{
			for (int x = 0; x < nx; x++, k++)
			{
				dx = (float)(x - (int)nx / 2) / (float)nx;
				dy = (float)(y - (int)ny / 2) / (float)ny;
				a = cos_theta_2 / (2 * sigma_x_2) + sin_theta_2 / (2 * sigma_y_2);
				b = -sin_2_theta / (4 * sigma_x_2) + sin_2_theta / (4 * sigma_y_2);
				c = sin_theta_2 / (2 * sigma_x_2) + cos_theta_2 / (2 * sigma_y_2);
				Z = exp(-(a*dx*dx - 2 * b*dx*dy + c*dy*dy));
				kernel_data[(((y + 2 * ny - (int)ny / 2) % ny))*nx + ((x + 2 * nx - (int)nx / 2) % nx)] = Z;
			}
		}
		create(rotation_kernel_token_time, 1, ny, nx, kernel_data, false); // create time domain buffer for kernel, this can be removed after the frequency domain data is obtained
		fft(1, ny, nx, rotation_kernel_token_time, rotation_kernel_token_freq);
		remove(rotation_kernel_token_time);
	}
	virtual void compute_semblance(
		int win
		, int nz
		, int ny
		, int nx
		, Token transpose
		, Token numerator
		, Token denominator
		) = 0;
	void initialize_semblance(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, Token transpose_token
		, Token numerator_token_time
		, Token denominator_token_time
		, Token numerator_token_freq
		, Token denominator_token_freq
		)
	{
		create(numerator_token_time, nz, ny, nx); // create numerator array (time domain) {X,Y,Z}
		create(denominator_token_time, nz, ny, nx); // create denominator array (time domain) {X,Y,Z}
		int win = 1;
		compute_semblance(win, nz, ny, nx, transpose_token, numerator_token_time, denominator_token_time);
		fft(nz, ny, nx, numerator_token_time, numerator_token_freq);
		fft(nz, ny, nx, denominator_token_time, denominator_token_freq);
	}
	virtual void compute_transpose(
		int nx
		, int ny
		, int nz
		, Token input
		, Token transpose
		) = 0;

	template<typename A>
	bool createArray(
		Token token
		, std::size_t nx,std::size_t ny,std::size_t nz
		, float * dat
		, bool keep_data
		)
	{
		A * arr = new A(token, nx,ny,nz);
		if (put(token, arr))
		{
			// allocate 
			(dat) ? arr->allocate(dat, keep_data) : arr->allocate();
			return true;
		}
		else
		{
			// just reuse the one already there
			(dat) ? arr->set(dat) : arr->fill(0);
			if (dat&&!keep_data)delete dat;
			delete arr;
			return false;
		}
	}

	virtual void compute_convolution_rotation_kernel(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, Token rotation_kernel_token_freq
		, Token signal_token_freq
		, Token rotated_signal_token_freq
		) = 0;
	
	virtual void compute_shear(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, float shear_y
		, float shear_x
		, Token rotated_freq
		, Token sheared_freq
		) = 0;

	virtual void compute_shear_time(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, float shear_y
		, float shear_x
		, Token rotated_time
		, Token sheared_time
		) = 0;

	virtual void init_fft(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		) = 0;

	virtual void destroy_fft() = 0;

	virtual void inv_fft(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, Token sheared_freq
		, Token sheared_time
		) = 0;

	virtual void compute_zsmooth(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, Token data
		) = 0;

	virtual void compute_fault_likelihood(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, Token num
		, Token den
		, Token data
		) = 0;

  virtual void compute_thin(
    std::size_t nz
    , std::size_t ny
    , std::size_t nx
    , float threshold
    , Token fh
    , Token th
    , Token thin
    ) = 0;

  ////////////////////////////////////////////////////////////////////////
  //
  // D_u f(x,y) = (a,b) . (dz/dx,dz/dy) = a dz/dx + b dz/dy
  // D^2_u f(x,y) = a^2 (d^2z/dx2) + 2ab (d^2z/dydx) + b^2 (d^2z/dy^2)
  //
  // D_u F(kx,ky) = (a kx + b ky) F_(kx,ky)
  // D^2_u F(kx,ky) = (a^2 kx^2 + 2 a b kx ky + b^2 ky^2) F_(kx,ky)
  //
  ////////////////////////////////////////////////////////////////////////
  //void init_first_directional_derivative_kernel(
  //  std::size_t nz
  //  , std::size_t ny
  //  , std::size_t nx
  //  , float theta
  //  , Token dat
  //  , Token th
  //  , Token deriv
  //  )
  //{
  //      	MESSAGE_ASSERT(
  //      		rotation_kernel_token_time.type == GPU
  //      		,  "argument type mismatch"
  //      		);
  //      	MESSAGE_ASSERT(
  //      		rotation_kernel_token_freq.type == FREQ_GPU
  //      		, "argument type mismatch"
  //      		);
  //      	cufftComplex * kernel_data = new cufftComplex[nx*ny];
  //      	float cos_theta = cos(theta);
  //      	float sin_theta = sin(theta);
  //      	for (int y = 0, k = 0; y < ny; y++)
  //      	{
  //      		for (int x = 0; x < nx; x++, k++)
  //      		{

  //      			kernel_data[(((y + 2 * ny - (int)ny / 2) % ny))*nx + ((x + 2 * nx - (int)nx / 2) % nx)].x = 0;
  //      			kernel_data[(((y + 2 * ny - (int)ny / 2) % ny))*nx + ((x + 2 * nx - (int)nx / 2) % nx)].y = ((float)x/nx)*cos_theta + ((float)y/ny)*sin_theta;
  //      		}
  //      	}
  //      	create(rotation_kernel_token_time, 1, ny, nx, kernel_data, false); // create time domain buffer for kernel, this can be removed after the frequency domain data is obtained
  //      	fft(1, ny, nx, rotation_kernel_token_time, rotation_kernel_token_freq);
  //      	remove(rotation_kernel_token_time);
  //}

  ////////////////////////////////////////////////////////////////////////
  //
  // D_u f(x,y) = (a,b) . (dz/dx,dz/dy) = a dz/dx + b dz/dy
  // D^2_u f(x,y) = a^2 (d^2z/dx2) + 2ab (d^2z/dydx) + b^2 (d^2z/dy^2)
  //
  // D_u F(kx,ky) = (a kx + b ky) F_(kx,ky)
  // D^2_u F(kx,ky) = (a^2 kx^2 + 2 a b kx ky + b^2 ky^2) F_(kx,ky)
  //
  ////////////////////////////////////////////////////////////////////////
  //void init_second_directional_derivative_kernel(
  //  std::size_t nz
  //  , std::size_t ny
  //  , std::size_t nx
  //  , float theta
  //  , Token dat
  //  , Token th
  //  , Token deriv
  //  )
  //{
  //      	MESSAGE_ASSERT(
  //      		rotation_kernel_token_time.type == GPU
  //      		,  "argument type mismatch"
  //      		);
  //      	MESSAGE_ASSERT(
  //      		rotation_kernel_token_freq.type == FREQ_GPU
  //      		, "argument type mismatch"
  //      		);
  //      	cufftComplex * kernel_data = new cufftComplex[nx*ny];
  //      	float cos_theta = cos(theta);
  //      	float sin_theta = sin(theta);
  //      	float cos_theta_2 = cos_theta*cos_theta;
  //      	float sin_theta_2 = sin_theta*sin_theta;
  //      	float sin_cos_theta = sin_theta*cos_theta;
  //      	for (int y = 0, k = 0; y < ny; y++)
  //      	{
  //      		for (int x = 0; x < nx; x++, k++)
  //      		{

  //      			kernel_data[(((y + 2 * ny - (int)ny / 2) % ny))*nx + ((x + 2 * nx - (int)nx / 2) % nx)].x = ((float)x/nx)*((float)x/nx)*cos_theta_2 + 2*((float)x/nx)*((float)y/ny)*sin_cos_theta + ((float)y/ny)*((float)y/ny)*sin_theta_2;
  //      			kernel_data[(((y + 2 * ny - (int)ny / 2) % ny))*nx + ((x + 2 * nx - (int)nx / 2) % nx)].y = 0;
  //      		}
  //      	}
  //      	create(rotation_kernel_token_time, 1, ny, nx, kernel_data, false); // create time domain buffer for kernel, this can be removed after the frequency domain data is obtained
  //      	fft(1, ny, nx, rotation_kernel_token_time, rotation_kernel_token_freq);
  //      	remove(rotation_kernel_token_time);
  //}

	virtual void update_maximum(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
    , float theta
		, Token fh
		, Token optimum_fh
		, Token optimum_th
		//, Token optimum_phi
		) = 0;

	void compute_fault_likelihood_time(
		int nz
		, int ny
		, int nx
		, float shear_y
		, float shear_x
    , float theta
		, Token rotated_numerator_token_time
		, Token rotated_denominator_token_time
		, Token sheared_numerator_token_time
		, Token sheared_denominator_token_time/*this guys is recycled internally, which is fine, since it is repopulated with fresh data with each shear iteration*/
		, Token fault_likelihood_token/*time domain {Z,Y,X}*/
		, Token optimal_fault_likelihood_token
		, Token optimal_theta_token
		//, Token optimal_phi_token
		)
	{
		// shear numerator
		compute_shear_time(nz, ny, nx, shear_y, shear_x, rotated_numerator_token_time, sheared_numerator_token_time);
		compute_zsmooth(nz, ny, nx, sheared_numerator_token_time);
		// shear denominator
		compute_shear_time(nz, ny, nx, shear_y, shear_x, rotated_numerator_token_time, sheared_denominator_token_time);
		compute_zsmooth(nz, ny, nx, sheared_denominator_token_time);
		// S = N/D
		// F = 1-S^8
		create(fault_likelihood_token, nz, ny, nx);
		compute_fault_likelihood(nz, ny, nx, sheared_numerator_token_time, sheared_denominator_token_time, fault_likelihood_token);
		// update max F
		update_maximum(nz, ny, nx
      , theta
			, fault_likelihood_token
			, optimal_fault_likelihood_token
			, optimal_theta_token
			//, optimal_phi_token
			);
		remove(fault_likelihood_token);
	}

	void init_shear(
		int nz
		, int ny
		, int nx
		, Token fault_likelihood_token
		, Token sheared_numerator_token_freq
		, Token sheared_denominator_token_freq
		)
	{
		Token num_token("tmp_numerator",GPU,nz,ny,nx);
		create(num_token, nz, ny, nx);
		Token den_token("tmp_denominator",GPU,nz,ny,nx);
		create(den_token, nz, ny, nx);
		create(fault_likelihood_token, nz, ny, nx);
		Token out_cplx_num_token(sheared_numerator_token_freq.name + "_cplx",FREQ_GPU,nz,ny,nx);
		create(out_cplx_num_token, nz, ny, nx, NULL, true);
		Token out_cplx_den_token(sheared_denominator_token_freq.name + "_cplx",FREQ_GPU,nz,ny,nx);
		create(out_cplx_den_token, nz, ny, nx, NULL, true);
	}

	void compute_fault_likelihood(
		int nz
		, int ny
		, int nx
		, float shear_y
		, float shear_x
    , float theta
		, Token rotated_numerator_token_freq
		, Token rotated_denominator_token_freq
		, Token sheared_numerator_token_freq
		, Token sheared_denominator_token_freq/*this guys is recycled internally, which is fine, since it is repopulated with fresh data with each shear iteration*/
		, Token fault_likelihood_token/*time domain {Z,Y,X}*/
		, Token optimal_fault_likelihood_token
		, Token optimal_theta_token
		//, Token optimal_phi_token
		)
	{
		// shear numerator
		compute_shear(nz, ny, nx, shear_y, shear_x, rotated_numerator_token_freq, sheared_numerator_token_freq);
		Token num_token("tmp_numerator",FREQ_GPU,nz,ny,nx);
		inv_fft(nz, ny, nx, sheared_numerator_token_freq, num_token);
		compute_zsmooth(nz, ny, nx, num_token);
		// shear denominator
		compute_shear(nz, ny, nx, shear_y, shear_x, rotated_denominator_token_freq, sheared_denominator_token_freq);
		Token den_token("tmp_denominator",FREQ_GPU,nz,ny,nx);
		inv_fft(nz, ny, nx, sheared_denominator_token_freq, den_token);
		compute_zsmooth(nz, ny, nx, den_token);
		// S = N/D
		// F = 1-S^8
		compute_fault_likelihood(nz, ny, nx, num_token, den_token, fault_likelihood_token);
		//remove(num_token);
		//remove(den_token);
		// update max F
		update_maximum(nz, ny, nx
      , theta
			, fault_likelihood_token
			, optimal_fault_likelihood_token
			, optimal_theta_token
			//, optimal_phi_token
			);
		//remove(fault_likelihood_token);
	}

};

#endif

