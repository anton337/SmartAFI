#ifndef AFIfunctor_h
#define AFIfunctor_h

#include "fft.h"

#include "transpose.h"

#include "semblance.h"

#include "shear.h"

#include "display_update.h"

#include "array_verify.h"

void zsmooth(
	float scalar
	, std::size_t num_z
	, std::size_t num_y
	, std::size_t num_x
	, float * input
	)
{
	
	float * prev_arr = &input[0];
	float * curr_arr = &input[num_y*num_x];
	for (int k = 0, size = (num_z - 1)*num_y*num_x; k < size; k++)
	{
		curr_arr[k] += scalar*prev_arr[k];
	}
	for (int k = 0, ind = num_x*num_y*(num_z-1)-1, size = (num_z - 1)*num_y*num_x; k < size; k++, ind--)
	{
		prev_arr[ind] += scalar*curr_arr[ind];
	}
	
}

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
		std::size_t num_x
		, std::size_t num_y
		, std::size_t num_z
		, std::size_t pad
		, float * input // {X,Y,Z}
		)
	{
		// input = {X,Y,Z}
		tile_display_update->update("input",num_x, num_y, num_z, input);
		// transpose: {X,Y,Z} -> {Z,Y,X}
		float * data_zyx = new float[num_x*num_y*num_z]; // {Z,Y,X}
		transpose_constY(num_x, num_y, num_z, input, data_zyx);
		tile_display_update->update1("zyx", num_z, num_y, num_x, data_zyx);
		// semblance on: {Z,Y,X}
		float * num = new float[num_x*num_y*num_z]; // {Z,Y,X}
		float * den = new float[num_x*num_y*num_z]; // {Z,Y,X}
		int win = 1;
		semblance_structure_oriented(win, num_z, num_y, num_x, data_zyx, num, den);
		if (tile_display_update->comprehensive)tile_display_update->update("numerator", num_z, num_y, num_x, num);
		if (tile_display_update->comprehensive)tile_display_update->update("denominator", num_z, num_y, num_x, den);
		
		std::size_t num_sheared_y = num_y + 2*64;
		std::size_t num_sheared_x = num_x + 2*64;

		float * rotation_kernel = new float[num_x*num_y*num_z]; // {Z,Y,X}
		float * rotation_kernel_display = new float[num_x*num_y*num_z]; // {Z,Y,X}
		float * num_rotation = new float[num_x*num_y*num_z]; // {Z,Y,X}
		float * den_rotation = new float[num_x*num_y*num_z]; // {Z,Y,X}
		float * num_shear = new float[num_sheared_x * num_sheared_y * num_z]; // {Z,Y_sheared,X_sheared}
		float * den_shear = new float[num_sheared_x * num_sheared_y * num_z]; // {Z,Y_sheared,X_sheared}
		float * shear_semblance = new float[num_sheared_x * num_sheared_y * num_z]; // {Z,Y_sheared,X_sheared}
		float * semblance = new float[num_x * num_y * num_z]; // {Z,Y,X}
		float * semblance_optimum = new float[num_x * num_y * num_z]; // {Z,Y,X}
		float * fault_likelihood_optimum = new float[num_x * num_y * num_z]; // {Z,Y,X}

		for (int k = 0, size = num_x*num_y*num_z; k < size; k++)
		{
			semblance_optimum[k] = 1.0f;
			fault_likelihood_optimum[k] = 0.0f;
		}

		float s = 2.0;
		for (float theta = -M_PI; theta <= 0; theta += M_PI / 32)
		{
			memset(rotation_kernel, 0, num_x*num_y);
			memset(rotation_kernel_display, 0, num_x*num_y);
			{
				float sigma_x_2 = 0.005f/(s*s);
				float sigma_y_2 = 0.000002f*10.0f;
				float a, b, c;
				float cos_theta = cos(theta);
				float cos_theta_2 = cos_theta*cos_theta;
				float sin_theta = sin(theta);
				float sin_theta_2 = sin_theta*sin_theta;
				float sin_2_theta = sin(2 * theta);
				float Z;
				float dx, dy;
				for (int y = 0, k = 0; y < num_y; y++)
				{
					for (int x = 0; x < num_x; x++, k++)
					{
						dx = (float)(x - (int)num_x / 2) / (float)num_x;
						dy = (float)(y - (int)num_y / 2) / (float)num_y;
						a = cos_theta_2 / (2 * sigma_x_2) + sin_theta_2 / (2 * sigma_y_2);
						b = -sin_2_theta / (4 * sigma_x_2) + sin_2_theta / (4 * sigma_y_2);
						c = sin_theta_2 / (2 * sigma_x_2) + cos_theta_2 / (2 * sigma_y_2);
						Z = exp(-(a*dx*dx - 2 * b*dx*dy + c*dy*dy));
						rotation_kernel[(((y + 2 * num_y - (int)num_y / 2) % num_y))*num_x + ((x + 2 * num_x - (int)num_x / 2) % num_x)] = Z;
						rotation_kernel_display[k] = Z;
					}
				}
			}
			tile_display_update->update2("rotation kernel", 1, num_y, num_x, rotation_kernel_display);
			// convolve: rotation_kernel * num
			
			compute_convolution_2d_slices_fast_b_c2c(num_z, num_y, num_x, num, rotation_kernel, num_rotation);
			if (tile_display_update->comprehensive)tile_display_update->update("numerator rotated", num_z, num_y, num_x, num_rotation);

			// convolve: rotation_kernel * den
			
			compute_convolution_2d_slices_fast_b_c2c(num_z, num_y, num_x, den, rotation_kernel, den_rotation);
			if (tile_display_update->comprehensive)tile_display_update->update("denominator rotated", num_z, num_y, num_x, den_rotation);

			float shear_extend = 0.1f;
			//for (float shear = -shear_extend; shear <= shear_extend; shear += shear_extend/4.0f)
			float shear = 0.0f;
			{
				std::cout << "theta:" << theta << "   shear:" << shear << std::endl;

				float shear_y = shear*cos(theta);
				float shear_z = -shear*sin(theta);

				shear_2d(FORWARD, LINEAR/*FFT*/, 1, num_z, num_y, num_x, num_sheared_y, num_sheared_x, shear_y, shear_z, num_rotation, num_shear);
				if (tile_display_update->comprehensive)tile_display_update->update("numerator sheared", num_z, num_sheared_y, num_sheared_x, num_shear);
				shear_2d(FORWARD, LINEAR/*FFT*/, 1, num_z, num_y, num_x, num_sheared_y, num_sheared_x, shear_y, shear_z, den_rotation, den_shear);
				if (tile_display_update->comprehensive)tile_display_update->update("denominator sheared", num_z, num_sheared_y, num_sheared_x, den_shear);

				float scalar = 0.9f;
				zsmooth(scalar, num_z, num_sheared_y, num_sheared_x, num_shear);
				if (tile_display_update->comprehensive)tile_display_update->update("numerator zsmooth", num_z, num_sheared_y, num_sheared_x, num_shear);
				zsmooth(scalar, num_z, num_sheared_y, num_sheared_x, den_shear);
				if (tile_display_update->comprehensive)tile_display_update->update("denomiantor zsmooth", num_z, num_sheared_y, num_sheared_x, num_shear);

				semblance_div(num_z, num_sheared_y, num_sheared_x, shear_semblance, num_shear, den_shear);
				if (tile_display_update->comprehensive)tile_display_update->update("semblance div", num_z, num_sheared_y, num_sheared_x, shear_semblance);

				shear_2d(BACKWARD, LINEAR/*FFT*/, 0, num_z, num_y, num_x, num_sheared_y, num_sheared_x, shear_y, shear_z, semblance, shear_semblance);
				if (tile_display_update->comprehensive)tile_display_update->update("semblance", num_z, num_y, num_x, semblance);

				//for (int k = 0, size = num_x*num_y*num_z; k < size; k++)
				//{
				//	semblance_optimum[k] = (semblance[k] < semblance_optimum[k]) ? semblance[k] : semblance_optimum[k];
				//}

				float val;
				float fh;
				for (int z = 0, k = 0; z < num_z; z++)
				for (int y = 0; y < num_y; y++)
				for (int x = 0; x < num_x; x++,k++)
				//if (x >= pad && x + pad < num_x)
				//if (y >= pad && y + pad < num_y)
				//if (z >= pad && z + pad < num_z)
				{
					val = semblance[k];
					val *= val;
					val *= val;
					val *= val;
					fh = 1.0f - val;
					fault_likelihood_optimum[k] = (fh>fault_likelihood_optimum[k])?fh:fault_likelihood_optimum[k];
				}

				tile_display_update->update("fault_likelihood_optimum", num_z, num_y, num_x, fault_likelihood_optimum);

			}

		}

		transpose_constY(num_z, num_y, num_x, fault_likelihood_optimum, input);
		
		tile_display_update->clear();
		tile_display_update->clear1();
		tile_display_update->clear2();

		delete[] data_zyx;
		delete[] semblance;
		delete[] semblance_optimum;
		delete[] fault_likelihood_optimum;
		delete[] shear_semblance;
		delete[] num_shear;
		delete[] den_shear;
		delete[] num_rotation;
		delete[] den_rotation;
		delete[] rotation_kernel;
		delete[] rotation_kernel_display;
		delete[] num;
		delete[] den;
	}
};

#endif
