#ifndef compute_manager_h
#define compute_manager_h

#include "compute_device.h"

#include "gpu_device.h"

#include "cpu_device.h"

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
				std::stringstream ss_name;
				ss_name << "GPU Device:" << i << "-" << ind;
				device.push_back(new GPUDevice(ss_name.str(),ind));
			}
		}
	}
	void addCPUDevices(int copies)
	{
		for (int i = 0; i < copies; i++)
		{
			device.push_back(new CPUDevice("CPU Device:"+i));
		}
	}
};

#endif

