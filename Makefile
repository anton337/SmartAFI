
all: cu_data_transfer.ptx cu_fft.ptx cu_semblance.ptx cu_shear.ptx cu_smooth.ptx cu_transpose.ptx cu_thin.ptx
	echo ${JAVA_HOME}
	echo "/usr/local/cuda"
	#gcc -w -fPIC -O3 -g0 main.cpp -o smartafi -lglut -lGL -lGLU -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -lm -lstdc++ -lboost_system -lboost_thread -lfftw3f -L/usr/local/cuda/lib64 -lcuda -lcufft -lcudart > log 2>&1; 
	gcc -w -fPIC -O0 -g3 main.cpp -o smartafi -lglut -lGL -lGLU -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -lm -lstdc++ -lboost_system -lboost_thread -lfftw3f -L/usr/local/cuda/lib64 -lcuda -lcufft -lcudart > log 2>&1; 
	./smartafi

cu_data_transfer.ptx:
	nvcc -w -Wno-deprecated-gpu-targets -ptx cu_data_transfer.cu

cu_fft.ptx:
	nvcc -w -Wno-deprecated-gpu-targets -ptx cu_fft.cu

cu_semblance.ptx:
	nvcc -w -Wno-deprecated-gpu-targets -ptx cu_semblance.cu

cu_shear.ptx:
	nvcc -w -Wno-deprecated-gpu-targets -ptx cu_shear.cu

cu_smooth.ptx:
	nvcc -w -Wno-deprecated-gpu-targets -ptx cu_smooth.cu

cu_transpose.ptx:
	nvcc -w -Wno-deprecated-gpu-targets -ptx cu_transpose.cu

cu_thin.ptx:
	nvcc -w -Wno-deprecated-gpu-targets -ptx cu_thin.cu

clean:
	rm -f *.o *.ptx smartafi log

