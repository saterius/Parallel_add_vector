#include<stdio.h>

__global__ void parallel_vector_add(int* d_a, int* d_b, int* d_c, int* d_n) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < *d_n) {
		printf("I am anout to compute c[%d].\n", i);
		d_c[i] = d_a[i] + d_b[i];
	}
	else {
		printf("I am thread #%d, and doing nothing.\n", i);
	}
}

int main()
{
	//allocate and initialize host memory
	int n;
	scanf("%d", &n);
	int h_a[n];
	int h_b[n];
	int h_c[n];

	for (int i = 0; i < n; i++) {
		h_a[i] = i;
		h_b[i] = n - i;
	}

	//Part 1
	//allocate device memory for a, b, and c
	//copy a and b to device memory
	int *d_a, *d_b, *d_c, *d_n;
	cudaMalloc((void **) &d_a, n*sizeof(int));
	cudaMalloc((void **) &d_b, n*sizeof(int));
	cudaMalloc((void **) &d_c, n*sizeof(int));
	cudaMalloc((void **) &d_n, sizeof(int));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(d_a, &h_a, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &h_b, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);

	//Part 2
	//kernel launch code which let the device performs the actual vector addition
	if (n % 512)
		int amountBlock = (n/512) + 1;
	else
		int amountBlock = n/512;
	parallel_vector_add<<<amountBlock, 512>>>(d_a, d_b, d_c, d_n);
	cudaDeviceSynchronize();

	//Part 3
	//copy c to host memory
	cudaMemcpy(&h_c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	//free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	for (int i = 0; i < n; i++) {
		printf("%d ", h_c[i]);
	}

	printf("\ntime used = %f\n", milliseconds);
}
