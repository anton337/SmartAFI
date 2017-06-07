//#define BOOST_ALL_DYN_LINK

#define _USE_MATH_DEFINES
#include <cmath>

#include <GL/glut.h>

#include <jni.h>

#include <cuda.h>

#include <cufft.h>

#include <helper_cuda_drvapi.h>

#include <cuda_profiler_api.h>

#include <cuda_runtime.h>

#include "AFI.h"

#include "unit_test.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#ifndef _checkCudaErrors
#define _checkCudaErrors(err)  ___checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void ___checkCudaErrors(CUresult err, const char *file, const int line)
{
	if (CUDA_SUCCESS != err)
	{
		fprintf(stdout, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
			err, getCudaDrvErrorString(err), file, line);
		char ch;
		std::cin >> ch;
	}
}
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		char ch;
		std::cin >> ch;
		if (abort) exit(code);
	}
}

GLfloat xRotated, yRotated, zRotated;

float Distance = 5;

DisplayUpdate * d_global_update = new DisplayUpdate();
DisplayUpdate * d_local_update = new DisplayUpdate();

void init(void)
{
	glEnable(GL_DEPTH_TEST);
	glClearColor(0, 0, 0, 0);
}

void drawstring(float x, float y, float z, const char *str)
{
	const char *c;
	glRasterPos3f(x, y, z);
	for (c = str; *c != '\0'; c++)
	{
		glColor3f(0.0, 0.0, 0.0);
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
	}
}

void DrawLocal(void)
{
	glMatrixMode(GL_MODELVIEW);
	// clear the drawing buffer.
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glColor3f(1, 1, 1);
	d_local_update->_mutex->lock();
	std::stringstream ss;
	ss << d_local_update->message;
	ss << ":" << d_local_update->layer_index << "/" << d_local_update->layers;
	drawstring(3, 3, -10, ss.str().c_str());
	d_local_update->_mutex->unlock();
	glTranslatef(0.0, 0.0, -Distance);
	glRotatef(xRotated, 1.0, 0.0, 0.0);
	// rotation about Y axis
	glRotatef(yRotated, 0.0, 1.0, 0.0);
	// rotation about Z axis
	glRotatef(zRotated, 0.0, 0.0, 1.0);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	d_local_update->draw();
	glFlush();
}

void animation(void)
{
	DrawLocal();
	boost::this_thread::sleep(boost::posix_time::milliseconds(500));
}

void Draw(void)
{
	DrawLocal();
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:exit(0);
	case 'p':d_local_update->pause = false; break;
	case 'c':d_local_update->continuous_processing = !d_local_update->continuous_processing; break;
	case 'w':d_local_update->increment_layer(); break;
	case 's':d_local_update->decrement_layer(); break;
	case 'v':d_local_update->set_comprehensive(); break;
	case 'q':Distance /= 1.01f; break;
	case 'a':Distance *= 1.01f; break;
	}
}

void reshape(int x, int y)
{
	if (y == 0 || x == 0) return;  //Nothing is visible then, so return
	//Set a new projection matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//Angle of view:40 degrees
	//Near clipping plane distance: 0.5
	//Far clipping plane distance: 20.0

	gluPerspective(40.0, (GLdouble)x / (GLdouble)y, 0.5, 20.0);
	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, x, y);  //Use the whole window for rendering
}

JNIEXPORT void JNICALL Java_AFI_AFI
(JNIEnv * env, jobject obj, jlong nx, jlong ny, jlong nz, jfloatArray arr)
{
	printf("hello world\n");

	std::size_t _nx = (std::size_t)nx;
	std::size_t _ny = (std::size_t)ny;
	std::size_t _nz = (std::size_t)nz;

	std::cout << "nx:" << nx << "   ny:" << ny << "   nz:" << nz << std::endl;

	std::size_t len = (std::size_t)env->GetArrayLength(arr);
	std::cout << "len:" << len << std::endl;

	float * _arr = new float[len]; 

	memset(_arr, 0, len);

	
	jfloat * tmp = new jfloat[800];
	
	float val;
	int size = 800;
	std::cout << "get elements" << std::endl;
	for (jsize k(0); k+size < len;k+=size)
	{
		//std::cout << "k=" << k << std::endl;
		env->GetFloatArrayRegion(arr, k, size, tmp);
		for (int i = 0; i < 800; i++)
		{
			//std::cout << i << " " << (float)tmp[i] << std::endl;
			val = (float)tmp[i];
			//val = (val>10000) ? 10000 : (val < -10000) ? -10000 : val;
			_arr[(int)k + i] = val;
		}
	}

	float max_val = -10000000;
	float min_val = 10000000;
	for (std::size_t k(0); k < len; k++)
	{
		_arr[k] = (_arr[k]>max_val) ? _arr[k] : max_val;
		_arr[k] = (_arr[k]<min_val) ? _arr[k] : min_val;
	}
	float fact = 1.0f / (max_val-min_val);
	for (std::size_t k(0); k < len; k++)
	{
		_arr[k] *= fact;
	}

	//std::cout << "release" << std::endl;
	//env->ReleaseFloatArrayElements(arr, _arr, 0);
	
	std::cout << "done memory swap" << std::endl;
	UnitTest * u_test = new UnitTest();
	u_test -> operator () (d_global_update,d_local_update,_nx,_ny,_nz,_arr);

	int argc = 1;
	char** argv = new char*[1];
	argv[0] = "AFI";

	glutInit(&argc, argv);
	//we initizlilze the glut. functions
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(1200, 800);
	glutCreateWindow(argv[0]);
	init();
	glutDisplayFunc(Draw);
	glutReshapeFunc(reshape);
	//Set the function for the animation.
	glutIdleFunc(animation);
	glutKeyboardFunc(keyboard);
	glutMainLoop();

	
}

// CUDA STUFF

// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error)
{
	switch (error)
	{
	case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";

	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";

	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";

	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";

	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";

	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";

	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";

	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";

	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";

	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";
	}

	return "<unknown>";
}

inline void cufftErrchk(cufftResult error, bool abort = true)
{
	if (error != CUFFT_SUCCESS)
	{
		fprintf(stderr, "cufft assert: %s\n", _cudaGetErrorEnum(error));
		char ch;
		std::cin >> ch;
		if (abort) exit(error);
	}
}

template<typename T>
void compute_shear(CUfunction fun,int ind,T * input_data, T * output_data, int nx, int ny, int nz, T shift_x, T shift_y)
{
	// shear each z slice by constant vector
	// the data order is {Z,Y,X}, where Z is the slowest dimension, and X is the fastest
	//
	
	cufftHandle plan;
	cufftComplex * input_mem = new cufftComplex[nx*ny*nz];
	cufftComplex * kernel_mem = new cufftComplex[nx*ny*nz];
	cufftComplex * output_mem = new cufftComplex[nx*ny*nz];
	float center_shift_x = 0;
	float center_shift_y = 0;
	T sz;
	for (int z = 0, k = 0; z<nz; z++)
	{
		sz = (T)(z) / (nz);
		for (int y = 0; y < ny; y++)
			for (int x = 0; x < nx; x++, k++)
			{
				kernel_mem[k].x =  cos((2 * M_PI*(x*center_shift_x + y*center_shift_y + x*sz*shift_x + y*sz*shift_y)) / ny);
				kernel_mem[k].y = -sin((2 * M_PI*(x*center_shift_x + y*center_shift_y + x*sz*shift_x + y*sz*shift_y)) / ny);
			}
	}
	for (int z = 0, k = 0; z<nz; z++)
		for (int y = 0; y<ny; y++)
			for (int x = 0; x<nx; x++, k++)
			{
				input_mem[k].x = input_data[k];
				input_mem[k].y = 0;
			}
	cufftComplex * input;
	cufftComplex * output;
	cufftComplex * fk_input;
	cufftComplex * fk_kernel;
	cufftComplex * fk_output;
	cudaMalloc((void**)&input, sizeof(cufftComplex)*nx*ny*nz);
	cudaMalloc((void**)&output, sizeof(cufftComplex)*nx*ny*nz);
	cudaMalloc((void**)&fk_input, sizeof(cufftComplex)*nx*ny*nz);
	cudaMalloc((void**)&fk_kernel, sizeof(cufftComplex)*nx*ny*nz);
	cudaMalloc((void**)&fk_output, sizeof(cufftComplex)*nx*ny*nz);
	int sizex = nx; 
	int sizey = ny;
	int sizez = nz;
	int n[2] = { ny, nx };
	cufftErrchk(cufftPlanMany(&plan, 2, n,
		NULL, 1, 0,
		NULL, 1, 0,
		CUFFT_C2C, nz)
		);
	int block_x = BLOCK_SIZE;
	int block_y = BLOCK_SIZE;
	int block_z = 1;
	dim3 block = dim3(block_x, block_y, block_z);
	//dim3 grid = dim3((nz + block_z - 1) / block_z, (ny + block_y - 1) / block_y, (nx + block_x - 1) / block_x);
	dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
	std::cout <<  grid.x << " " <<  grid.y << " " <<  grid.z << std::endl;
	std::cout << block.x << " " << block.y << " " << block.z << std::endl;
	void *args[6] = { &sizex, &sizey, &sizez, &fk_input, &fk_kernel, &fk_output };
	gpuErrchk(cudaMemcpy(input, input_mem, sizeof(cufftComplex)*nx*ny*nz, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(fk_kernel, kernel_mem, sizeof(cufftComplex)*nx*ny*nz, cudaMemcpyHostToDevice));
	//for (int k = 0; k < 1000; k++)
	{
		cufftErrchk(cufftExecC2C(plan, input, fk_input, CUFFT_FORWARD));
		//cudaProfilerStart();
		//Hadamard<<<grid,block>>>(nx,nz,ny,ny*nz,fk_input,fk_kernel,fk_output);
		//Hadamard << <(nx*ny*nz + 31) / 32, 32 >> >(nx*ny*nz, nx*ny*nz, fk_input, fk_kernel, fk_output);
		//cudaProfilerStop();
		// new CUDA 4.0 Driver API Kernel launch call
		//CUresult CUDAAPI cuLaunchKernel(
		//	CUfunction f,
		//	unsigned int gridDimX,
		//	unsigned int gridDimY,
		//	unsigned int gridDimZ,
		//	unsigned int blockDimX,
		//	unsigned int blockDimY,
		//	unsigned int blockDimZ,
		//	unsigned int sharedMemBytes,
		//	CUstream hStream,
		//	void **kernelParams,
		//	void **extra);

		_checkCudaErrors(cuLaunchKernel(fun, 
			//1,1,1,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));

		_checkCudaErrors(cuCtxSynchronize());

		cufftErrchk(cufftExecC2C(plan, fk_output, output, CUFFT_INVERSE));
		//cufftErrchk(cufftExecC2C(plan, fk_input, output, CUFFT_INVERSE));
	}
	gpuErrchk(cudaMemcpy(output_mem, output, sizeof(cufftComplex)*nx*ny*nz, cudaMemcpyDeviceToHost));
	verify("fk output:", nx, ny, nz, output_mem);
	verify_data("fk output:", nx, ny, nz, nx, output_mem,true);
	cufftDestroy(plan);
	T inv = 1.0f / (nx*ny);
	for (int z = 0, k = 0; z<nz; z++)
		for (int y = 0; y<ny; y++)
			for (int x = 0; x<nx; x++, k++)
			{
				output_data[k] = output_mem[k].x*inv;
			}
	cudaFree(input);
	cudaFree(output);
	cudaFree(fk_input);
	cudaFree(fk_kernel);
	cudaFree(fk_output);
	delete[] input_mem;
	delete[] kernel_mem;
	delete[] output_mem;
	
}

template<typename T>
void compute_rotation_convolution(CUfunction fun, int ind, T * input_data, T * output_data, int nx, int ny, int nz, T theta)
{
	// shear each z slice by constant vector
	// the data order is {Z,Y,X}, where Z is the slowest dimension, and X is the fastest
	//
	cufftHandle plan;
	cufftHandle plan1;
	cufftComplex * input_mem = new cufftComplex[nx*ny*nz];
	cufftComplex * kernel_mem = new cufftComplex[nx*ny];
	cufftComplex * output_mem = new cufftComplex[nx*ny*nz];
	/*
	float s = 2.0;
	for (float theta = -M_PI; theta <= 0; theta += M_PI / 32)
	{
		memset(rotation_kernel, 0, num_x*num_y);
		memset(rotation_kernel_display, 0, num_x*num_y);
		{
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
	}
	*/
	{
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
		for (int y = 0; y < ny; y++)
			for (int x = 0; x < nx; x++, k++)
			{
				dx = (float)(x - (int)nx / 2) / (float)nx;
				dy = (float)(y - (int)ny / 2) / (float)ny;
				a = cos_theta_2 / (2 * sigma_x_2) + sin_theta_2 / (2 * sigma_y_2);
				b = -sin_2_theta / (4 * sigma_x_2) + sin_2_theta / (4 * sigma_y_2);
				c = sin_theta_2 / (2 * sigma_x_2) + cos_theta_2 / (2 * sigma_y_2);
				Z = exp(-(a*dx*dx - 2 * b*dx*dy + c*dy*dy));
				kernel_mem[(((y + 2 * ny - (int)ny / 2) % ny))*nx + ((x + 2 * nx - (int)nx / 2) % nx)].x = Z;
				kernel_mem[(((y + 2 * ny - (int)ny / 2) % ny))*nx + ((x + 2 * nx - (int)nx / 2) % nx)].y = 0;
			}
	}
	for (int z = 0, k = 0; z<nz; z++)
		for (int y = 0; y<ny; y++)
			for (int x = 0; x<nx; x++, k++)
			{
				input_mem[k].x = input_data[k];
				input_mem[k].y = 0;
			}
	cufftComplex * input;
	cufftComplex * kernel;
	cufftComplex * output;
	cufftComplex * fk_input;
	cufftComplex * fk_kernel;
	cufftComplex * fk_output;
	cudaMalloc((void**)&input, sizeof(cufftComplex)*nx*ny*nz);
	cudaMalloc((void**)&kernel, sizeof(cufftComplex)*nx*ny);
	cudaMalloc((void**)&output, sizeof(cufftComplex)*nx*ny*nz);
	cudaMalloc((void**)&fk_input, sizeof(cufftComplex)*nx*ny*nz);
	cudaMalloc((void**)&fk_kernel, sizeof(cufftComplex)*nx*ny);
	cudaMalloc((void**)&fk_output, sizeof(cufftComplex)*nx*ny*nz);
	int sizex = nx;
	int sizey = ny;
	int sizez = nz;
	int n[2] = { ny, nx };
	cufftErrchk(cufftPlanMany(&plan, 2, n,
		NULL, 1, 0,
		NULL, 1, 0,
		CUFFT_C2C, nz)
		);
	cufftErrchk(cufftPlanMany(&plan1, 2, n,
		NULL, 1, 0,
		NULL, 1, 0,
		CUFFT_C2C, 1)
		);
	int block_x = BLOCK_SIZE;
	int block_y = BLOCK_SIZE;
	int block_z = 1;
	dim3 block = dim3(block_x, block_y, block_z);
	//dim3 grid = dim3((nz + block_z - 1) / block_z, (ny + block_y - 1) / block_y, (nx + block_x - 1) / block_x);
	dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
	std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
	std::cout << block.x << " " << block.y << " " << block.z << std::endl;
	void *args[6] = { &sizex, &sizey, &sizez, &fk_input, &fk_kernel, &fk_output };
	gpuErrchk(cudaMemcpy(input, input_mem, sizeof(cufftComplex)*nx*ny*nz, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(kernel, kernel_mem, sizeof(cufftComplex)*nx*ny, cudaMemcpyHostToDevice));
	//for (int k = 0; k < 1000; k++)
	{
		cufftErrchk(cufftExecC2C(plan, input, fk_input, CUFFT_FORWARD));
		cufftErrchk(cufftExecC2C(plan1, kernel, fk_kernel, CUFFT_FORWARD));
		//cudaProfilerStart();
		//Hadamard<<<grid,block>>>(nx,nz,ny,ny*nz,fk_input,fk_kernel,fk_output);
		//Hadamard << <(nx*ny*nz + 31) / 32, 32 >> >(nx*ny*nz, nx*ny*nz, fk_input, fk_kernel, fk_output);
		//cudaProfilerStop();
		// new CUDA 4.0 Driver API Kernel launch call
		//CUresult CUDAAPI cuLaunchKernel(
		//	CUfunction f,
		//	unsigned int gridDimX,
		//	unsigned int gridDimY,
		//	unsigned int gridDimZ,
		//	unsigned int blockDimX,
		//	unsigned int blockDimY,
		//	unsigned int blockDimZ,
		//	unsigned int sharedMemBytes,
		//	CUstream hStream,
		//	void **kernelParams,
		//	void **extra);

		_checkCudaErrors(cuLaunchKernel(fun,
			//1,1,1,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));

		_checkCudaErrors(cuCtxSynchronize());

		cufftErrchk(cufftExecC2C(plan, fk_output, output, CUFFT_INVERSE));
		//cufftErrchk(cufftExecC2C(plan, fk_input, output, CUFFT_INVERSE));
	}
	gpuErrchk(cudaMemcpy(output_mem, output, sizeof(cufftComplex)*nx*ny*nz, cudaMemcpyDeviceToHost));
	verify("fk output:", nx, ny, nz, output_mem);
	verify_data("fk output:", nx, ny, nz, nx, output_mem, true);
	cufftDestroy(plan);
	cufftDestroy(plan1);
	T inv = 1.0f / (nx*ny);
	for (int z = 0, k = 0; z<nz; z++)
		for (int y = 0; y<ny; y++)
			for (int x = 0; x<nx; x++, k++)
			{
				output_data[k] = output_mem[k].x*inv;
			}
	cudaFree(input);
	cudaFree(output);
	cudaFree(fk_input);
	cudaFree(fk_kernel);
	cudaFree(fk_output);
	delete[] input_mem;
	delete[] kernel_mem;
	delete[] output_mem;
}

void test_shear(CUfunction fun,int ind)
{
	std::cout << "begin: test shear" << std::endl;
	int nx = 64;
	int ny = 64;
	int nz = 64;
	float * in = new float[nx*ny*nz];
	float *out = new float[nx*ny*nz];
	for (std::size_t x = 0, k = 0; x < nx; x++)
		for (std::size_t y = 0; y < ny; y++)
			for (std::size_t z = 0; z < nz; z++, k++)
			{
				out[k] = 0;
				in[k] = 0;
				if (	pow((int)x - (int)nx / 2, 2)
					+	pow((int)y - (int)ny / 2, 2)
					+	pow((int)z - (int)nz / 2, 2)
					>	pow(nx / 2, 2)
					)
				{
					in[k] += 5.0f;
				}
			}
	float shift_x = 0.0f;
	float shift_y = 0.0f;
	d_local_update->update("in:", nx, ny, nz, in);
	compute_shear(fun, ind, in, out, nx, ny, nz, shift_x, shift_y);
	d_local_update->update("out:", nx, ny, nz, out);
	std::cout << "done: test shear" << std::endl;
}

void test_convolution_kernel(CUfunction fun, int ind)
{
	std::cout << "begin: test convolution kernel" << std::endl;
	int nx = 64;
	int ny = 64;
	int nz = 64;
	float * in = new float[nx*ny*nz];
	float *out = new float[nx*ny*nz];
	for (std::size_t x = 0, k = 0; x < nx; x++)
		for (std::size_t y = 0; y < ny; y++)
			for (std::size_t z = 0; z < nz; z++, k++)
			{
				out[k] = 0;
				in[k] = 0;
				if (pow((int)x - (int)nx / 2, 2)
					+ pow((int)y - (int)ny / 2, 2)
					+ pow((int)z - (int)nz / 2, 2)
			>	pow(nx / 2, 2)
			)
				{
					in[k] += 5.0f;
				}
			}
	float theta = M_PI/3;
	d_local_update->update("in:", nx, ny, nz, in);
	compute_rotation_convolution(fun, ind, in, out, nx, ny, nz, theta);
	d_local_update->update("out:", nx, ny, nz, out);
	std::cout << "done: test shear" << std::endl;
}

void test_gpu()
{
	CUdevice cuDevice;
	CUcontext cuContext;
	CUmodule cuModule;
	CUfunction cuFunction = 0;
	std::size_t totalGlobalMem;
	int major = 0, minor = 0;
	char deviceName[100];

	_checkCudaErrors(cuInit(0));

	int devices = 0;
	_checkCudaErrors(cuDeviceGetCount(&devices));
	printf("num devices:%d\n",devices);
	
	/*
	for (int ind = 0; ind < devices;ind++)
	{
		printf("device %d:\n",ind);
		_checkCudaErrors(cuDeviceGet(&cuDevice, ind));
		// get compute capabilities and the devicename
		_checkCudaErrors(cuDeviceComputeCapability(&major, &minor, cuDevice));
		_checkCudaErrors(cuDeviceGetName(deviceName, 256, cuDevice));
		printf("> GPU Device has SM %d.%d compute capability\n", major, minor);
		_checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, cuDevice));
		printf("  Total amount of global memory:     %llu bytes\n", (unsigned long long)totalGlobalMem);
		printf("  64-bit Memory Address:             %s\n", (totalGlobalMem > (unsigned long long)4 * 1024 * 1024 * 1024L) ? "YES" : "NO");
	
		_checkCudaErrors(cuDeviceGet(&cuDevice, ind));
		_checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));
		
		_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_fft.ptx"));
		_checkCudaErrors(cuModuleGetFunction(&cuFunction, cuModule, "Hadamard"));
		
		test_shear(cuFunction,ind);

		_checkCudaErrors(cuCtxDestroy(cuContext));
		
	}
	*/
	
	for (int ind = 0; ind < devices; ind++)
	{
		printf("device %d:\n", ind);
		_checkCudaErrors(cuDeviceGet(&cuDevice, ind));
		// get compute capabilities and the devicename
		_checkCudaErrors(cuDeviceComputeCapability(&major, &minor, cuDevice));
		_checkCudaErrors(cuDeviceGetName(deviceName, 256, cuDevice));
		printf("> GPU Device has SM %d.%d compute capability\n", major, minor);
		_checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, cuDevice));
		printf("  Total amount of global memory:     %llu bytes\n", (unsigned long long)totalGlobalMem);
		printf("  64-bit Memory Address:             %s\n", (totalGlobalMem >(unsigned long long)4 * 1024 * 1024 * 1024L) ? "YES" : "NO");

		_checkCudaErrors(cuDeviceGet(&cuDevice, ind));
		_checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));

		_checkCudaErrors(cuModuleLoad(&cuModule, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/7_CUDALibraries/smartAFI/x64/Release/cu_fft.ptx"));
		_checkCudaErrors(cuModuleGetFunction(&cuFunction, cuModule, "Hadamard_slice_kernel"));

		test_convolution_kernel(cuFunction, ind);

		_checkCudaErrors(cuCtxDestroy(cuContext));

	}

}

void quick_zsmooth_sanity_check()
{
	float dat[64];
	for (int i = 0; i < 64; i++)
	{
		dat[i] = 0;
	}
	int ind = 32;
	dat[ind] = 1;
	float a, b;
	b = 4;
	a = b - 1;
	float alpha = a/b;
	for (int i = 1; i < 64; i++)
	{
		dat[i] += dat[i - 1] * alpha;
	}
	float beta = a/b;
	for (int i = 64 - 2; i >= 0; i--)
	{
		dat[i] += dat[i + 1] * beta;
	}
	float factor = b*b / (a + b);
	int w = 5;
	for (int k = -w; k <= w; k++)
	{
		std::cout << ind+k << ") " << dat[ind + k]/factor << std::endl;
	}
}

enum TYPE {FREQ_CPU,CPU,FREQ_GPU,GPU};

class DataArray
{
	std::string name;
	std::size_t size;
	TYPE type;
public:
	DataArray(std::string _name,std::size_t _size,TYPE _type)
		: name(_name)
		, size(_size)
		, type(_type)
	{

	}
	std::string get_name()
	{
		return name;
	}
	std::size_t get_size()
	{
		return size;
	}
	TYPE get_type()
	{
		return type;
	}
	virtual void allocate(float * arr = NULL, bool keep_data = false) = 0;
	virtual void put(void *) = 0;
	virtual void set(float *) = 0;
	virtual void fill(float) = 0;
	virtual void * get() = 0;
	virtual void destroy() = 0;
};

class freqGPUArray : public DataArray
{
	cufftComplex * data;
public:
	freqGPUArray(std::string name
				, std::size_t size)
		: DataArray(name, size,FREQ_GPU)
	{

	}
	void put(void * _data)
	{
		std::cout << "freq GPU put not implemented yet." << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}
	void * get()
	{
		return (void*)data;
	}
	void allocate(float * arr = NULL, bool keep_data = false)
	{
		//std::cout << "cudaMalloc:" << get_name() << std::endl;
		gpuErrchk(cudaMalloc((void**)&data, sizeof(cufftComplex)*get_size()));
		if (arr)
		{
			cufftComplex * tmp = new cufftComplex[get_size()];
			for (int i = 0; i < get_size(); i++)
			{
				tmp[i].x = arr[i];
				tmp[i].y = 0;
			}
			if (!keep_data)
			{
				delete[] arr; // don't forget
			}
			gpuErrchk(cudaMemcpy(data, tmp, sizeof(cufftComplex)*get_size(), cudaMemcpyHostToDevice));
			delete[] tmp;
		}
	}
	void set(float * arr)
	{
		std::cout << "freq GPU set not implemented yet." << std::endl;
		delete[] arr; // don't forget
		char ch; std::cin >> ch;
		exit(0);
	}
	void fill(float dat)
	{
		// can't figure it out, just leave it empty
		// cufftComplex data is randomly initialized
	}
	void destroy()
	{
		gpuErrchk(cudaFree(data));
	}
};

class GPUArray : public DataArray
{
	float * data;
public:
	GPUArray(std::string name
			,std::size_t size)
		: DataArray(name, size,GPU)
	{

	}
	void put(void * _data)
	{
		std::cout << "GPU put not implemented yet." << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}
	void * get()
	{
		return (void*)data;
	}
	void allocate(float * arr = NULL, bool keep_data = false)
	{
		//std::cout << "cudaMalloc:" << get_name() << std::endl;
		gpuErrchk(cudaMalloc((void**)&data, sizeof(float)*get_size()));
		if (arr)
		{
			gpuErrchk(cudaMemcpy(data, arr, sizeof(float)*get_size(), cudaMemcpyHostToDevice));
			if (!keep_data)
			{
				delete[] arr;
			}
		}
		else
		{
			gpuErrchk(cudaMemset(data,0,sizeof(float)*get_size()));
		}
	}
	void set(float * arr)
	{
		std::cout << "GPU set not implemented yet." << std::endl;
		delete[] arr; // don't forget
		char ch; std::cin >> ch;
		exit(0);
	}
	void fill(float arr)
	{
		// do nothing for now
		// data is randomly initialized
	}
	void destroy()
	{
		cudaFree(data);
	}
};

class freqCPUArray : public DataArray
{
	fftwf_complex * data;
public:
	freqCPUArray(std::string name
			,std::size_t size)
		: DataArray(name,size,FREQ_CPU)
	{

	}
	void put(void * _data)
	{
		data = (fftwf_complex*)_data;
	}
	void * get()
	{
		return (void*)data;
	}	
	void allocate(float * arr = NULL, bool keep_data = false)
	{
		std::cout << "freq CPU allocate not implemented yet." << std::endl;
		delete[] arr; // don't forget
		char ch; std::cin >> ch;
		exit(0);
	}
	void set(float * arr)
	{
		std::cout << "freq CPU allocate not implemented yet." << std::endl;
		delete[] arr; // don't forget
		char ch; std::cin >> ch;
		exit(0);
	}
	void fill(float arr)
	{
		std::cout << "freq CPU fill not implemented yet." << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}
	void destroy()
	{
		delete[] data;
	}
};

class CPUArray : public DataArray
{
	float * data;
public:
	CPUArray(std::string name
			,std::size_t size)
		: DataArray(name,size,CPU)
	{

	}
	void put(void * _data)
	{
		data = (float*)_data;
	}
	void * get()
	{
		return (void*)data;
	}
	void allocate(float * arr = NULL,bool keep_data = false)
	{
		if (arr)
		{
			data = arr;
		}
		else
		{
			data = new float[get_size()];
		}
	}
	void set(float * arr)
	{
		for (int i = 0, size = get_size(); i < size; i++)
		{
			data[i] = arr[i];
		}
		delete[] arr; // don't forget
		char ch; std::cin >> ch;
		exit(0);
	}
	void fill(float arr)
	{
		std::cout << "CPU fill not implemented yet." << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}
	void destroy()
	{
		delete[] data;
	}
};

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

class CPUDevice : public ComputeDevice
{
	void fft(int nz, int ny, int nx, std::string in_name, std::string out_name)
	{
		std::cout << "cpu fft not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	bool create(std::string name, int nx, int ny, int nz, float * dat = NULL, bool freq = false, bool keep_data = false)
	{
		return (freq) ? createArray<freqCPUArray>(name, nx*ny*nz, dat, keep_data)
			: createArray<CPUArray>(name, nx*ny*nz, dat, keep_data);
	}

	void compute_semblance(int win, int nz, int ny, int nx, std::string transpose, std::string numerator, std::string denominator)
	{
		std::cout << "cpu compute semblance not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void compute_transpose(int nx, int ny, int nz, std::string input, std::string transpose)
	{
		std::cout << "cpu transpose not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void compute_convolution_rotation_kernel(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, std::string rotation_kernel_name_freq
		, std::string signal_name_freq
		, std::string rotated_signal_name_freq
		)
	{
		std::cout << "cpu rotation not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void compute_shear(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, float shear_y
		, float shear_x
		, std::string rotated_freq
		, std::string sheared_freq
		)
	{
		std::cout << "cpu compute shear not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void compute_shear_time(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, float shear_y
		, float shear_x
		, std::string rotated_time
		, std::string sheared_time
		)
	{
		std::cout << "cpu compute shear time not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void inv_fft(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, std::string sheared_freq
		, std::string sheared_time
		)
	{
		std::cout << "cpu inv fft not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void init_fft(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		)
	{
		std::cout << "cpu init fft not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void destroy_fft()
	{
		std::cout << "cpu destroy fft not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void compute_zsmooth(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, std::string data
		)
	{
		std::cout << "cpu compute zsmooth not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void compute_fault_likelihood(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, std::string num
		, std::string den
		, std::string data
		)
	{
		std::cout << "cpu compute fault likelihood not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

	void update_maximum(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, std::string fh
		, std::string optimum_fh
		//, std::string optimum_th
		//, std::string optimum_phi
		)
	{
		std::cout << "cpu update maximum not implemented yet" << std::endl;
		char ch; std::cin >> ch;
		exit(0);
	}

};

class GPUDevice : public ComputeDevice
{
	std::size_t ind;

	std::size_t totalGlobalMem;

	CUcontext cuContext;

	CUdevice cuDevice;
	CUmodule cuModule;

	char deviceName[100];

	cufftHandle plan;
	cufftHandle plan1;

public:
	GPUDevice(std::size_t _ind)
	{
		ind = _ind;
		int major = 0, minor = 0;
		printf("device %d:\n", ind);
		_checkCudaErrors(cuDeviceGet(&cuDevice, ind));
		// get compute capabilities and the devicename
		_checkCudaErrors(cuDeviceComputeCapability(&major, &minor, cuDevice));
		_checkCudaErrors(cuDeviceGetName(deviceName, 256, cuDevice));
		printf("> GPU Device has SM %d.%d compute capability\n", major, minor);
		_checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, cuDevice));
		printf("  Total amount of global memory:     %llu bytes\n", (unsigned long long)totalGlobalMem);
		printf("  64-bit Memory Address:             %s\n", (totalGlobalMem >(unsigned long long)4 * 1024 * 1024 * 1024L) ? "YES" : "NO");
		create_context();
		load_convolution_rotation_kernel();
		load_shear();
		load_shear_time();
		load_zsmooth();
		load_semblance();
		load_semblance_div();
		load_semblance_max();
		load_transpose();
		load_data_transfer_real_cplx();
		load_data_transfer_cplx_real();
	}
	~GPUDevice()
	{
		destroy_context();
	}
	
	bool create(std::string name, int nx, int ny, int nz, float * dat = NULL, bool freq = false, bool keep_data = false)
	{
		return (freq)	? createArray<freqGPUArray>(name, nx*ny*nz, dat, keep_data) 
						: createArray<GPUArray>(name, nx*ny*nz, dat, keep_data);
	}
private:
	CUfunction convolution_rotation_kernel;
	CUfunction shear;
	CUfunction shear_time;
	CUfunction zsmooth;
	CUfunction semblance;
	CUfunction semblance_div;
	CUfunction semblance_max;
	CUfunction transpose_constY;
	CUfunction data_transfer_real_cplx;
	CUfunction data_transfer_cplx_real;
	void create_context()
	{
		_checkCudaErrors(cuDeviceGet(&cuDevice, ind));
		_checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));
	}
	void destroy_context()
	{
		_checkCudaErrors(cuCtxDestroy(cuContext));
	}
	void set_context()
	{
		_checkCudaErrors(cuCtxSetCurrent(cuContext));
	}
	void load_convolution_rotation_kernel()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_fft.ptx"));
		_checkCudaErrors(cuModuleGetFunction(&convolution_rotation_kernel, cuModule, "Hadamard_slice_kernel"));
	}
	void load_shear()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_shear.ptx"));
		_checkCudaErrors(cuModuleGetFunction(&shear, cuModule, "Shear"));
	}
	void load_shear_time()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_shear.ptx"));
		_checkCudaErrors(cuModuleGetFunction(&shear_time, cuModule, "ShearTimeDomain"));
	}
	void load_zsmooth()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_smooth.ptx"));
		_checkCudaErrors(cuModuleGetFunction(&zsmooth, cuModule, "zSmooth"));
	}
	void load_semblance_div()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_semblance.ptx"));
		_checkCudaErrors(cuModuleGetFunction(&semblance_div, cuModule, "SemblanceDiv"));
	}
	void load_semblance_max()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_semblance.ptx"));
		_checkCudaErrors(cuModuleGetFunction(&semblance_max, cuModule, "SemblanceMax"));
	}
	void load_semblance()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_semblance.ptx"));
		_checkCudaErrors(cuModuleGetFunction(&semblance, cuModule, "Semblance"));
	}
	void load_transpose()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_transpose.ptx"));
		_checkCudaErrors(cuModuleGetFunction(&transpose_constY, cuModule, "transpose_constY"));
	}
	void load_data_transfer_real_cplx()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_data_transfer.ptx"));
		_checkCudaErrors(cuModuleGetFunction(&data_transfer_real_cplx, cuModule, "data_transfer_real_cplx"));
	}
	void load_data_transfer_cplx_real()
	{
		set_context();
		_checkCudaErrors(cuModuleLoad(&cuModule, "cu_data_transfer.ptx"));
		_checkCudaErrors(cuModuleGetFunction(&data_transfer_cplx_real, cuModule, "data_transfer_cplx_real"));
	}

public:
	void compute_convolution_rotation_kernel(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, std::string rotated_kernel_name_freq
		, std::string signal_name_freq
		, std::string rotated_signal_name_freq
		)
	{
		set_context();
		int sizex = nx;
		int sizey = ny;
		int sizez = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		cufftComplex * fk_input  = (cufftComplex*)get(signal_name_freq);
		cufftComplex * fk_kernel = (cufftComplex*)get(rotated_kernel_name_freq);
		cufftComplex * fk_output = (cufftComplex*)get(rotated_signal_name_freq);
		void *args[6] = { &sizex, &sizey, &sizez, &fk_input, &fk_kernel, &fk_output };
		{
			_checkCudaErrors(cuLaunchKernel(convolution_rotation_kernel,
				grid.x, grid.y, grid.z,
				block.x, block.y, block.z,
				0,
				NULL, args, NULL));
			_checkCudaErrors(cuCtxSynchronize());
		}
	}
	
	void compute_shear(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, float shear_y
		, float shear_x
		, std::string rotated_freq
		, std::string sheared_freq
		)
	{
		set_context();
		int sizex = nx;
		int sizey = ny;
		int sizez = nz;
		float SHEAR_Y = shear_y;
		float SHEAR_X = shear_x;
		float CENTER_SHEAR_Y = 0;
		float CENTER_SHEAR_X = 0;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		cufftComplex * fk_rotated = (cufftComplex*)get(rotated_freq);
		cufftComplex * fk_sheared = (cufftComplex*)get(sheared_freq);
		void *args[9] = { &CENTER_SHEAR_Y, &CENTER_SHEAR_X, &SHEAR_Y, &SHEAR_X, &sizez, &sizey, &sizex, &fk_rotated, &fk_sheared };
		{
			_checkCudaErrors(cuLaunchKernel(shear,
				grid.x, grid.y, grid.z,
				block.x, block.y, block.z,
				0,
				NULL, args, NULL));
			_checkCudaErrors(cuCtxSynchronize());
		}
	}

	void compute_shear_time(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, float shear_y
		, float shear_x
		, std::string rotated_time
		, std::string sheared_time
		)
	{
		set_context();
		int sizex = nx;
		int sizey = ny;
		int sizez = nz;
		float SHEAR_Y = shear_y;
		float SHEAR_X = shear_x;
		float CENTER_SHEAR_Y = 0;
		float CENTER_SHEAR_X = 0;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		float * rotated = (float*)get(rotated_time);
		float * sheared = (float*)get(sheared_time);
		void *args[9] = { &CENTER_SHEAR_Y, &CENTER_SHEAR_X, &SHEAR_Y, &SHEAR_X, &sizez, &sizey, &sizex, &rotated, &sheared };
		{
			_checkCudaErrors(cuLaunchKernel(shear_time,
				grid.x, grid.y, grid.z,
				block.x, block.y, block.z,
				0,
				NULL, args, NULL));
			_checkCudaErrors(cuCtxSynchronize());
		}
	}

	void init_fft(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		)
	{
		std::cout << "init fft" << std::endl;
		{
			int n[2] = { ny, nx };
			cufftErrchk(cufftPlanMany(&plan, 2, n,
				NULL, 1, 0,
				NULL, 1, 0,
				CUFFT_C2C, nz)
				);
		}
		{
			int n[2] = { ny, nx };
			cufftErrchk(cufftPlanMany(&plan1, 2, n,
				NULL, 1, 0,
				NULL, 1, 0,
				CUFFT_C2C, 1)
				);
		}
		std::cout << "done init fft" << std::endl;
	}

	void destroy_fft()
	{
		cufftErrchk(cufftDestroy(plan));
		cufftErrchk(cufftDestroy(plan1));
	}

	void inv_fft(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, std::string sheared_freq
		, std::string sheared_time
		)
	{
		set_context();
		
		cufftComplex * a_fft = (cufftComplex*)get(sheared_freq);
		std::string out_cplx_name = sheared_freq + "_cplx";
		cufftComplex * a_cplx_out = (cufftComplex*)get(out_cplx_name);
		
		if (nz==1)
			cufftErrchk(cufftExecC2C(plan1, a_fft, a_cplx_out, CUFFT_INVERSE));
		else
			cufftErrchk(cufftExecC2C(plan, a_fft, a_cplx_out, CUFFT_INVERSE));

		float * a_out = (float*)get(sheared_time);
		int NX = nx;
		int NY = ny;
		int NZ = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		void *args[5] = { &NX, &NY, &NZ, &a_cplx_out, &a_out };
		_checkCudaErrors(cuLaunchKernel(data_transfer_cplx_real,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));
		_checkCudaErrors(cuCtxSynchronize());

		
	}

	void compute_zsmooth(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, std::string data
		)
	{
		set_context();
		float * a_data = (float*)get(data);
		float alpha = 0.8;
		int NX = nx;
		int NY = ny;
		int NZ = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		void *args[5] = { &NZ, &NY, &NX, &alpha, &a_data };
		_checkCudaErrors(cuLaunchKernel(zsmooth,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));
		_checkCudaErrors(cuCtxSynchronize());
	}

	void compute_fault_likelihood(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, std::string num
		, std::string den
		, std::string data
		)
	{
		set_context();
		float * a_num = (float*)get(num);
		float * a_den = (float*)get(den);
		float * a_data = (float*)get(data);
		int NX = nx;
		int NY = ny;
		int NZ = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		void *args[6] = { &NZ, &NY, &NX, &a_num, &a_den, &a_data };
		_checkCudaErrors(cuLaunchKernel(semblance_div,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));
		_checkCudaErrors(cuCtxSynchronize());
	}

	void update_maximum(
		std::size_t nz
		, std::size_t ny
		, std::size_t nx
		, std::string fh
		, std::string optimum_fh
		//, std::string optimum_th
		//, std::string optimum_phi
		)
	{
		set_context();
		float * a_data = (float*)get(fh);
		float * a_optimum = (float*)get(optimum_fh);
		int NX = nx;
		int NY = ny;
		int NZ = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		void *args[6] = { &NZ, &NY, &NX, &a_data, &a_optimum };
		_checkCudaErrors(cuLaunchKernel(semblance_max,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));
		_checkCudaErrors(cuCtxSynchronize());
	}
	void compute_semblance(int win, int nz, int ny, int nx, std::string transpose, std::string numerator, std::string denominator)
	{
		set_context();
		int WIN = win;
		int NX = nx;
		int NY = ny;
		int NZ = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		float * a_data = (float*)get(transpose);
		float * a_num = (float*)get(numerator);
		float * a_den = (float*)get(denominator);
		void *args[7] = { &WIN, &NX, &NY, &NZ, &a_data, &a_num, &a_den };
		_checkCudaErrors(cuLaunchKernel(semblance,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));
		_checkCudaErrors(cuCtxSynchronize());
		remove(transpose);
	}
	void compute_transpose(int nx,int ny,int nz,std::string input,std::string transpose)
	{
		set_context();
		int NX = nx;
		int NY = ny;
		int NZ = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		float * a_input = (float*)get(input);
		//verifyGPU("GPU input", nx, ny, nz, a_input,true);
		float * a_transpose = (float*)get(transpose);
		//verifyGPU("GPU transpose before", nx, ny, nz, a_transpose, true);
		void *args[5] = { &NX, &NY, &NZ, &a_input, &a_transpose };
		_checkCudaErrors(cuLaunchKernel(transpose_constY,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));
		_checkCudaErrors(cuCtxSynchronize());
		//verifyGPU("GPU transpose after", nx, ny, nz, a_transpose, true);
	}
	void fft(int nz, int ny, int nx, std::string in, std::string out)
	{
		set_context();
		float * a_input = (float*)get(in);
		std::string in_cplx_name = in + "_cplx";
		create(in_cplx_name, nz, ny, nx, NULL, true);
		create(out, nz, ny, nx, NULL, true);
		cufftComplex * in_cplx = (cufftComplex*)get(in_cplx_name);
		int NX = nx;
		int NY = ny;
		int NZ = nz;
		int block_x = BLOCK_SIZE;
		int block_y = BLOCK_SIZE;
		int block_z = 1;
		dim3 block = dim3(block_x, block_y, block_z);
		dim3 grid = dim3((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);
		void *args[5] = { &NX, &NY, &NZ, &a_input, &in_cplx };
		_checkCudaErrors(cuLaunchKernel(data_transfer_real_cplx,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0,
			NULL, args, NULL));
		_checkCudaErrors(cuCtxSynchronize());
		//verifyGPU("in cplx", nx, ny, nz, in_cplx, true);
		cufftComplex * a_fft = (cufftComplex*)get(out);
		//verifyGPU("a fft", nx, ny, nz, a_fft, true);
		
		if (nz == 1)
			cufftErrchk(cufftExecC2C(plan1, in_cplx, a_fft, CUFFT_FORWARD));
		else
			cufftErrchk(cufftExecC2C(plan, in_cplx, a_fft, CUFFT_FORWARD));

		//verifyGPU("a fft after", nx, ny, nz, a_fft, true);
		//_checkCudaErrors(cuCtxSynchronize());
		remove(in_cplx_name);
		remove(in);
	}
	
};

class ComputeManager
{
	std::vector<ComputeDevice*> device;
public:
	std::vector<ComputeDevice*> create_compute_devices(std::size_t num_cpu_copies, std::size_t num_gpu_copies)
	{
		addCPUDevices(num_cpu_copies);
		if (num_gpu_copies > 0)
		{
			initGPU();
			addGPUDevices(num_gpu_copies);
		}
		return device;
	}
	void destroy()
	{

	}
private:
	void initGPU()
	{
		_checkCudaErrors(cuInit(0));
	}
	void addGPUDevices(int copies)
	{
		int devices = 0;
		_checkCudaErrors(cuDeviceGetCount(&devices));
		printf("num devices:%d\n", devices);
		for (int i = 0; i < copies; i++)
		{
			for (int ind = 0; ind < devices; ind++)
			{
				device.push_back(new GPUDevice(ind));
			}
		}
	}
	void addCPUDevices(int copies)
	{
		for (int i = 0; i < copies; i++)
		{
			device.push_back(new CPUDevice());
		}
	}
};

void test_gpu_afi()
{
	
	int nx = 256;
	int ny = 256;
	int nz = 256;
	float * in = new float[nx*ny*nz];
	float *out = new float[nx*ny*nz];
	for (std::size_t x = 0, k = 0; x < nx; x++)
	{
		for (std::size_t y = 0; y < ny; y++)
		{
			for (std::size_t z = 0; z < nz; z++, k++)
			{
				out[k] = 0;
				in[k] = 0;
				if (pow((int)x - (int)nx / 2, 2)
					+ pow((int)y - (int)ny / 2, 2)
					+ pow((int)z - (int)nz / 2, 2)
					> pow(nx / 2, 2)
					)
				{
					in[k] = k%10;
				}
			}
		}
	}

	ComputeManager * c_manager = new ComputeManager();

	int num_cpu_copies = 0;
	int num_gpu_copies = 1;
	std::vector<ComputeDevice*> device = c_manager->create_compute_devices(num_cpu_copies,num_gpu_copies);

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

	for (int ind = 0; ind < device.size(); ind++)
	{
		std::cout << "device-" << ind << std::endl;
		ComputeDevice * d = device[ind];

		// input = {X,Y,Z}
		//tile_display_update->update("input", num_x, num_y, num_z, input);
		d->create(input_name, nx, ny, nz, in, false, true); // allocate input array (time domain) {X,Y,Z}

		// transpose: {X,Y,Z} -> {Z,Y,X}
		//float * data_zyx = new float[num_x*num_y*num_z]; // {Z,Y,X}
		d->create(transpose_name, nz, ny, nx); // create transpose array (time domain) {Z,Y,X}
		d->compute_transpose(nx, ny, nz, input_name, transpose_name);

		//transpose_constY(num_x, num_y, num_z, input, data_zyx);
		//tile_display_update->update1("zyx", num_z, num_y, num_x, data_zyx);
		// semblance on: {Z,Y,X}
		//float * num = new float[num_x*num_y*num_z]; // {Z,Y,X}
		//float * den = new float[num_x*num_y*num_z]; // {Z,Y,X}
		//d->create(numerator_name, nz, ny, nx); // create numerator array (time domain) {X,Y,Z}
		//d->create(denominator_name, nz, ny, nx); // create denominator array (time domain) {X,Y,Z}

		//int win = 1;
		//semblance_structure_oriented(win, num_z, num_y, num_x, data_zyx, num, den);
		//int win = 1;
		//d->compute_semblance(win, nz, ny, nx, transpose_name, numerator_name, denominator_name);
		//if (tile_display_update->comprehensive)tile_display_update->update("numerator", num_z, num_y, num_x, num);
		//if (tile_display_update->comprehensive)tile_display_update->update("denominator", num_z, num_y, num_x, den);

		d->init_fft(nz, ny, nx);

		d->initialize_semblance(nz,ny,nx,transpose_name,numerator_name,denominator_name,numerator_name_freq,denominator_name_freq);

		//std::size_t num_sheared_y = num_y + 2 * 64;
		//std::size_t num_sheared_x = num_x + 2 * 64;
		//std::size_t num_sheared_y = ny + 2 * 64;
		//std::size_t num_sheared_x = nx + 2 * 64;

		//float * rotation_kernel = new float[num_x*num_y*1]; // {1,Y,X}
		//float * rotation_kernel_display = new float[num_x*num_y*1]; // {1,Y,X}
		//float * num_rotation = new float[num_x*num_y*num_z]; // {Z,Y,X}
		//float * den_rotation = new float[num_x*num_y*num_z]; // {Z,Y,X}
		//float * num_shear = new float[num_sheared_x * num_sheared_y * num_z]; // {Z,Y_sheared,X_sheared}
		//float * den_shear = new float[num_sheared_x * num_sheared_y * num_z]; // {Z,Y_sheared,X_sheared}
		//float * shear_semblance = new float[num_sheared_x * num_sheared_y * num_z]; // {Z,Y_sheared,X_sheared}
		//float * semblance = new float[num_x * num_y * num_z]; // {Z,Y,X}
		//float * semblance_optimum = new float[num_x * num_y * num_z]; // {Z,Y,X}
		//float * fault_likelihood_optimum = new float[num_x * num_y * num_z]; // {Z,Y,X}

		//for (int k = 0, size = num_x*num_y*num_z; k < size; k++)
		//{
		//	semblance_optimum[k] = 1.0f;
		//	fault_likelihood_optimum[k] = 0.0f;
		//}

		d->create(optimal_fault_likelihood_name, nz, ny, nx);
		d->create(output_fault_likelihood_name, nx, ny, nz);

		d->create(rotated_numerator_name_freq, nz, ny, nx, NULL, true);
		d->create(rotated_denominator_name_freq, nz, ny, nx, NULL, true);

		d->create(rotated_numerator_name_time, nz, ny, nx, NULL, true);
		d->create(rotated_denominator_name_time, nz, ny, nx, NULL, true);

		d->create(sheared_numerator_name_freq, nz, ny, nx, NULL, true);
		d->create(sheared_denominator_name_freq, nz, ny, nx, NULL, true);

		d->create(sheared_numerator_name_time, nz, ny, nx, NULL, true);
		d->create(sheared_denominator_name_time, nz, ny, nx, NULL, true);

		d->init_shear(nz, ny, nx, fault_likelihood_name, sheared_numerator_name_freq, sheared_denominator_name_freq);

		//float s = 2.0;
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
		//	memset(rotation_kernel, 0, num_x*num_y);
		//	memset(rotation_kernel_display, 0, num_x*num_y);
		//	{
		//		float sigma_x_2 = 0.005f / (s*s);
		//		float sigma_y_2 = 0.000002f*10.0f;
		//		float a, b, c;
		//		float cos_theta = cos(theta);
		//		float cos_theta_2 = cos_theta*cos_theta;
		//		float sin_theta = sin(theta);
		//		float sin_theta_2 = sin_theta*sin_theta;
		//		float sin_2_theta = sin(2 * theta);
		//		float Z;
		//		float dx, dy;
		//		for (int y = 0, k = 0; y < num_y; y++)
		//		{
		//			for (int x = 0; x < num_x; x++, k++)
		//			{
		//				dx = (float)(x - (int)num_x / 2) / (float)num_x;
		//				dy = (float)(y - (int)num_y / 2) / (float)num_y;
		//				a = cos_theta_2 / (2 * sigma_x_2) + sin_theta_2 / (2 * sigma_y_2);
		//				b = -sin_2_theta / (4 * sigma_x_2) + sin_2_theta / (4 * sigma_y_2);
		//				c = sin_theta_2 / (2 * sigma_x_2) + cos_theta_2 / (2 * sigma_y_2);
		//				Z = exp(-(a*dx*dx - 2 * b*dx*dy + c*dy*dy));
		//				rotation_kernel[(((y + 2 * num_y - (int)num_y / 2) % num_y))*num_x + ((x + 2 * num_x - (int)num_x / 2) % num_x)] = Z;
		//				rotation_kernel_display[k] = Z;
		//			}
		//		}
		//	}
		//	tile_display_update->update2("rotation kernel", 1, num_y, num_x, rotation_kernel_display);
		// convolve: rotation_kernel * num
			
			d->compute_convolution_rotation_kernel(nz, ny, nx, ss_rotation_kernel_name_freq.str(), numerator_name_freq, rotated_numerator_name_freq);
			//d->compute_convolution_rotation_kernel(nz, ny, nx, ss_rotation_kernel_name_freq.str(), numerator_name_freq, rotated_numerator_name_freq);
			//d->inv_fft(nz, ny, nx, rotated_numerator_name_freq, rotated_numerator_name_time);

		//	compute_convolution_2d_slices_fast_b_c2c(num_z, num_y, num_x, num, rotation_kernel, num_rotation);
		//	if (tile_display_update->comprehensive)tile_display_update->update("numerator rotated", num_z, num_y, num_x, num_rotation);
			
		// convolve: rotation_kernel * den
			
			d->compute_convolution_rotation_kernel(nz, ny, nx, ss_rotation_kernel_name_freq.str(), denominator_name_freq, rotated_denominator_name_freq);
			//d->compute_convolution_rotation_kernel(nz, ny, nx, ss_rotation_kernel_name_freq.str(), denominator_name_freq, rotated_denominator_name_freq);
			//d->inv_fft(nz, ny, nx, rotated_denominator_name_freq, rotated_denominator_name_time);

		//	compute_convolution_2d_slices_fast_b_c2c(num_z, num_y, num_x, den, rotation_kernel, den_rotation);
		//	if (tile_display_update->comprehensive)tile_display_update->update("denominator rotated", num_z, num_y, num_x, den_rotation);

			float shear_extend = 0.1f;
			for (float shear = -shear_extend; shear <= shear_extend; shear += shear_extend/32.0f)
		//	float shear = 0.0f;
			{
				//std::cout << "theta:" << theta << "   shear:" << shear << std::endl;

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

				//d->compute_fault_likelihood_time(
				//	nz
				//	, ny
				//	, nx
				//	, shear_y
				//	, shear_x
				//	, rotated_numerator_name_time
				//	, rotated_denominator_name_time
				//	, sheared_numerator_name_time
				//	, sheared_denominator_name_time/*this guys is recycled internally, which is fine, since it is repopulated with fresh data with each shear iteration*/
				//	, fault_likelihood_name/*time domain {Z,Y,X}*/
				//	, optimal_fault_likelihood_name
				//	//, optimal_theta_name
				//	//, optimal_phi_name
				//	);


		//		shear_2d(FORWARD, LINEAR/*FFT*/, 1, num_z, num_y, num_x, num_sheared_y, num_sheared_x, shear_y, shear_z, num_rotation, num_shear);
		//		if (tile_display_update->comprehensive)tile_display_update->update("numerator sheared", num_z, num_sheared_y, num_sheared_x, num_shear);
		//		shear_2d(FORWARD, LINEAR/*FFT*/, 1, num_z, num_y, num_x, num_sheared_y, num_sheared_x, shear_y, shear_z, den_rotation, den_shear);
		//		if (tile_display_update->comprehensive)tile_display_update->update("denominator sheared", num_z, num_sheared_y, num_sheared_x, den_shear);

		//		float scalar = 0.9f;
		//		zsmooth(scalar, num_z, num_sheared_y, num_sheared_x, num_shear);
		//		if (tile_display_update->comprehensive)tile_display_update->update("numerator zsmooth", num_z, num_sheared_y, num_sheared_x, num_shear);
		//		zsmooth(scalar, num_z, num_sheared_y, num_sheared_x, den_shear);
		//		if (tile_display_update->comprehensive)tile_display_update->update("denomiantor zsmooth", num_z, num_sheared_y, num_sheared_x, num_shear);

		//		semblance_div(num_z, num_sheared_y, num_sheared_x, shear_semblance, num_shear, den_shear);
		//		if (tile_display_update->comprehensive)tile_display_update->update("semblance div", num_z, num_sheared_y, num_sheared_x, shear_semblance);

		//		shear_2d(BACKWARD, LINEAR/*FFT*/, 0, num_z, num_y, num_x, num_sheared_y, num_sheared_x, shear_y, shear_z, semblance, shear_semblance);
		//		if (tile_display_update->comprehensive)tile_display_update->update("semblance", num_z, num_y, num_x, semblance);

		//for (int k = 0, size = num_x*num_y*num_z; k < size; k++)
		//{
		//	semblance_optimum[k] = (semblance[k] < semblance_optimum[k]) ? semblance[k] : semblance_optimum[k];
		//}

		//		float val;
		//		float fh;
		//		for (int z = 0, k = 0; z < num_z; z++)
		//			for (int y = 0; y < num_y; y++)
		//				for (int x = 0; x < num_x; x++, k++)
		//if (x >= pad && x + pad < num_x)
		//if (y >= pad && y + pad < num_y)
		//if (z >= pad && z + pad < num_z)
		//				{
		//					val = semblance[k];
		//					val *= val;
		//					val *= val;
		//					val *= val;
		//					fh = 1.0f - val;
		//					fault_likelihood_optimum[k] = (fh>fault_likelihood_optimum[k]) ? fh : fault_likelihood_optimum[k];
		//				}

		//		tile_display_update->update("fault_likelihood_optimum", num_z, num_y, num_x, fault_likelihood_optimum);

			}

		}

		d->destroy_fft();

		//std::cout << "p1" << std::endl;
		d->compute_transpose(nz, ny, nx, optimal_fault_likelihood_name, output_fault_likelihood_name);
		//std::cout << "p2" << std::endl;
		//transpose_constY(num_z, num_y, num_x, fault_likelihood_optimum, input);

		//tile_display_update->clear();
		//tile_display_update->clear1();
		//tile_display_update->clear2();

		//delete[] data_zyx;
		//delete[] semblance;
		//delete[] semblance_optimum;
		//delete[] fault_likelihood_optimum;
		//delete[] shear_semblance;
		//delete[] num_shear;
		//delete[] den_shear;
		//delete[] num_rotation;
		//delete[] den_rotation;
		//delete[] rotation_kernel;
		//delete[] rotation_kernel_display;
		//delete[] num;
		//delete[] den;
	}
	//std::cout << "p3" << std::endl;
	c_manager->destroy();
	//std::cout << "p4" << std::endl;
	delete c_manager;
	//std::cout << "p5" << std::endl;
}

int main(int argc, char** argv)
{

	//quick_zsmooth_sanity_check();

	//char ch;
	//std::cin >> ch;

	test_gpu_afi();
	//test_gpu();
	//test_shear();
	
	//UnitTest * u_test = new UnitTest();
	//u_test -> operator () (d_global_update,d_local_update);

	/*
	glutInit(&argc, argv);
	//we initizlilze the glut. functions
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(1200, 800);
	glutCreateWindow(argv[0]);
	init();
	glutDisplayFunc(Draw);
	glutReshapeFunc(reshape);
	//Set the function for the animation.
	glutIdleFunc(animation);
	glutKeyboardFunc(keyboard);
	glutMainLoop();
	*/

	return 0;
}
