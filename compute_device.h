#ifndef compute_device_h
#define compute_device_h

#include "data_array.h"

#include "cuda_stuff.h"

class ComputeDevice
{
	typedef std::map<std::string, DataArray*> map_type;
	map_type data_array;
public:
	void* get(std::string name)
	{
		if (data_array.find(name) == data_array.end())
		{
			std::cout << "can't find array:" << name << std::endl;
			char ch;
			std::cin >> ch;
			exit(1);
		}
		return data_array[name]->get();
	}
	virtual bool create(std::string name, int nx, int ny, int nz, float * dat = NULL, bool freq = false, bool keep_data = false) = 0;
	bool put(std::string name, DataArray * dat)
	{
		if (data_array.find(name) == data_array.end())
		{
			destroy(name);
			data_array[name] = dat;
			return true;
		}
		else
		{
			if (dat->get_size() > data_array[name]->get_size()
				|| dat->get_type() != data_array[name]->get_type())
			{
				destroy(name);
				data_array[name] = dat;
				return true;
			}
		}
		return false;
	}
	void destroy(std::string name)
	{
		map_type::iterator it = data_array.find(name);
		if (it == data_array.end())
		{
			return;
		}
		it->second->destroy();
		data_array.erase(it);
	}
	void remove(std::string name)
	{
		destroy(name);
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
	virtual void fft(int nz, int ny, int nx, std::string in_name, std::string out_name) = 0;
	void initialize_rotation_kernel(
		int nx
		, int ny
		, float theta
		, std::string rotation_kernel_name_time
		, std::string rotation_kernel_name_freq
		)
	{
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
		create(rotation_kernel_name_time, 1, ny, nx, kernel_data, false); // create time domain buffer for kernel, this can be removed after the frequency domain data is obtained
		fft(1, ny, nx, rotation_kernel_name_time, rotation_kernel_name_freq);
		remove(rotation_kernel_name_time);
	}
	virtual void compute_semblance(int win, int nz, int ny, int nx, std::string transpose, std::string numerator, std::string denominator) = 0;
	void initialize_semblance(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, std::string transpose_name
		, std::string numerator_name_time
		, std::string denominator_name_time
		, std::string numerator_name_freq
		, std::string denominator_name_freq
		)
	{
		create(numerator_name_time, nz, ny, nx); // create numerator array (time domain) {X,Y,Z}
		create(denominator_name_time, nz, ny, nx); // create denominator array (time domain) {X,Y,Z}
		int win = 1;
		compute_semblance(win, nz, ny, nx, transpose_name, numerator_name_time, denominator_name_time);
		fft(nz, ny, nx, numerator_name_time, numerator_name_freq);
		fft(nz, ny, nx, denominator_name_time, denominator_name_freq);
	}
	virtual void compute_transpose(int nx, int ny, int nz, std::string input, std::string transpose) = 0;

	template<typename A>
	bool createArray(std::string name, std::size_t size, float * dat, bool keep_data)
	{
		A * arr = new A(name, size);
		if (put(name, arr))
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
		, std::string rotation_kernel_name_freq
		, std::string signal_name_freq
		, std::string rotated_signal_name_freq
		) = 0;
	
	virtual void compute_shear(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, float shear_y
		, float shear_x
		, std::string rotated_freq
		, std::string sheared_freq
		) = 0;

	virtual void compute_shear_time(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, float shear_y
		, float shear_x
		, std::string rotated_time
		, std::string sheared_time
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
		, std::string sheared_freq
		, std::string sheared_time
		) = 0;

	virtual void compute_zsmooth(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, std::string data
		) = 0;

	virtual void compute_fault_likelihood(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, std::string num
		, std::string den
		, std::string data
		) = 0;

	virtual void update_maximum(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, std::string fh
		, std::string optimum_fh
		//, std::string optimum_th
		//, std::string optimum_phi
		) = 0;

	void compute_fault_likelihood_time(
		int nz
		, int ny
		, int nx
		, float shear_y
		, float shear_x
		, std::string rotated_numerator_name_time
		, std::string rotated_denominator_name_time
		, std::string sheared_numerator_name_time
		, std::string sheared_denominator_name_time/*this guys is recycled internally, which is fine, since it is repopulated with fresh data with each shear iteration*/
		, std::string fault_likelihood_name/*time domain {Z,Y,X}*/
		, std::string optimal_fault_likelihood_name
		//, std::string optimal_theta_name
		//, std::string optimal_phi_name
		)
	{
		// shear numerator
		compute_shear_time(nz, ny, nx, shear_y, shear_x, rotated_numerator_name_time, sheared_numerator_name_time);
		compute_zsmooth(nz, ny, nx, sheared_numerator_name_time);
		// shear denominator
		compute_shear_time(nz, ny, nx, shear_y, shear_x, rotated_numerator_name_time, sheared_denominator_name_time);
		compute_zsmooth(nz, ny, nx, sheared_denominator_name_time);
		// S = N/D
		// F = 1-S^8
		create(fault_likelihood_name, nz, ny, nx);
		compute_fault_likelihood(nz, ny, nx, sheared_numerator_name_time, sheared_denominator_name_time, fault_likelihood_name);
		// update max F
		update_maximum(nz, ny, nx
			, fault_likelihood_name
			, optimal_fault_likelihood_name
			//, optimal_theta_name
			//, optimal_phi_name
			);
		remove(fault_likelihood_name);
	}

	void init_shear(int nz, int ny, int nx, std::string fault_likelihood_name, std::string sheared_numerator_name_freq, std::string sheared_denominator_name_freq)
	{
		std::string num_name = "tmp_numerator";
		create(num_name, nz, ny, nx);
		std::string den_name = "tmp_denominator";
		create(den_name, nz, ny, nx);
		create(fault_likelihood_name, nz, ny, nx);
		std::string out_cplx_num_name = sheared_numerator_name_freq + "_cplx";
		create(out_cplx_num_name, nz, ny, nx, NULL, true);
		std::string out_cplx_den_name = sheared_denominator_name_freq + "_cplx";
		create(out_cplx_den_name, nz, ny, nx, NULL, true);
	}

	void compute_fault_likelihood(
		int nz
		, int ny
		, int nx
		, float shear_y
		, float shear_x
		, std::string rotated_numerator_name_freq
		, std::string rotated_denominator_name_freq
		, std::string sheared_numerator_name_freq
		, std::string sheared_denominator_name_freq/*this guys is recycled internally, which is fine, since it is repopulated with fresh data with each shear iteration*/
		, std::string fault_likelihood_name/*time domain {Z,Y,X}*/
		, std::string optimal_fault_likelihood_name
		//, std::string optimal_theta_name
		//, std::string optimal_phi_name
		)
	{
		// shear numerator
		compute_shear(nz, ny, nx, shear_y, shear_x, rotated_numerator_name_freq, sheared_numerator_name_freq);
		std::string num_name = "tmp_numerator";
		inv_fft(nz, ny, nx, sheared_numerator_name_freq, num_name);
		compute_zsmooth(nz, ny, nx, num_name);
		// shear denominator
		compute_shear(nz, ny, nx, shear_y, shear_x, rotated_numerator_name_freq, sheared_denominator_name_freq);
		std::string den_name = "tmp_denominator";
		inv_fft(nz, ny, nx, sheared_denominator_name_freq, den_name);
		compute_zsmooth(nz, ny, nx, den_name);
		// S = N/D
		// F = 1-S^8
		compute_fault_likelihood(nz, ny, nx, num_name, den_name, fault_likelihood_name);
		//remove(num_name);
		//remove(den_name);
		// update max F
		update_maximum(nz, ny, nx
			, fault_likelihood_name
			, optimal_fault_likelihood_name
			//, optimal_theta_name
			//, optimal_phi_name
			);
		//remove(fault_likelihood_name);
	}

};

#endif

