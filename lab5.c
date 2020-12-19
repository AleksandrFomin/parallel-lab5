#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>

#if defined(_OPENMP)
#include "omp.h"
#else
struct timeval timeval;
double omp_get_wtime()
{
	gettimeofday(&timeval, NULL);
	return (double)timeval.tv_sec + (double)timeval.tv_usec / 1000000;
}

void omp_set_nested(int val)
{
	val = 1;
}
#endif

void fill_array(double *arr, int size, double left, double right, unsigned int *seedp)
{
	int i;

	#if defined(_OPENMP)
	#pragma omp parallel for default(none) private(i) shared(seedp, right, left, size, arr)
	#endif
	for (i = 0; i < size; i++) {
		unsigned int seed_i = i + *seedp;
		arr[i] = rand_r(&seed_i) / (double)RAND_MAX * (right - left) + left;
	}
}

void print_array(double *arr, int size)
{
	int i;
	printf("arr=[");
	for (i = 0; i < size; i++) {
		printf(" %f", arr[i]);
	}
	printf("]\n");
}

void map_m1(double *arr, int size)
{
	int i;

	#if defined(_OPENMP)
	#pragma omp parallel for default(none) private(i) shared(arr, size)
	#endif
	for (i = 0; i < size; i++) {
		arr[i] = tanh(arr[i]) - 1;
	}
}

void map_m2(double *arr, int size, double *arr_copy)
{
	int i;

	#if defined(_OPENMP)
	#pragma omp parallel for default(none) private(i) shared(arr, arr_copy, size)
	#endif
	for (i = 0; i < size; i++) {
		double prev = 0;
		if (i > 0)
			prev = arr_copy[i - 1];
		arr[i] = sqrt(exp(1.0) * (arr_copy[i] + prev));
	}
}

void copy_arr(double *src, int len, double *dst)
{
	int i;

	#if defined(_OPENMP)
	#pragma omp parallel for default(none) private(i) shared(src, dst, len)
	#endif
	for (i = 0; i < len; i++)
		dst[i] = src[i];
}

void apply_merge_func(double *m1, double *m2, int m2_len)
{
	int i;

	#if defined(_OPENMP)
	#pragma omp parallel for default(none) private(i) shared(m1, m2, m2_len)
	#endif
	for (i = 0; i < m2_len; i++) {
		m2[i] = fabs(m1[i] - m2[i]);
	}
}

void heapify(double *array, int n)
{
	int i,j,k;
	double item;
	for(k=1 ; k<n ; k++) {
		item = array[k];
		i = k;
		j = (i-1)/2;
		while( (i>0) && (item>array[j]) ) {
			array[i] = array[j];
			i = j;
			j = (i-1)/2;
		}
		array[i] = item;
	}
}

void adjust(double *array, int n)
{
	int i,j;
	double item;

	j = 0;
	item = array[j];
	i = 2*j+1;

	while(i<=n-1) {
		if(i+1 <= n-1)
			if(array[i] < array[i+1])
				i++;
		if(item < array[i]) {
			array[j] = array[i];
			j = i;
			i = 2*j+1;
		} else
			break;
	}
	array[j] = item;
}

void heapsort(double *array, int n)
{
	int i;
	double t;

	heapify(array,n);

	for(i=n-1 ; i>0 ; i--) {
		t = array[0];
		array[0] = array[i];
		array[i] = t;
		adjust(array,i);
	}
}

void mergeArrays(double *dst, double *arr1, double *arr2, int len1, int len2)
{
	int i, j, k;
	i = j = k = 0;
	for (i = 0; i < len1 && j < len2;) {
		if (arr1[i] < arr2[j]) {
			dst[k] = arr1[i];
			k++;
			i++;
		} else {
			dst[k] = arr2[j];
			k++;
			j++;
		}
	}
	while (i < len1) {
		dst[k] = arr1[i];
		k++;
		i++;
	}
	while (j < len2) {
		dst[k] = arr2[j];
		k++;
		j++;
	}
}

double min_not_null(double *arr, int len)
{
	int i;
	double min_val = DBL_MAX;
	for (i = 0; i < len; i++) {
		if (arr[i] < min_val && arr[i] > 0)
			min_val = arr[i];
	}
	return min_val;
}

double reduce(double *arr, int len)
{
	int i;
	double min_val = min_not_null(arr, len);
	double x = 0;

	#if defined(_OPENMP)
	#pragma omp parallel for default(none) private(i) shared(arr, len, min_val) reduction(+:x)
	#endif
	for (i = 0; i < len; i++) {
		if ((int)(arr[i] / min_val) % 2 == 0) {
			double sin_val = sin(arr[i]);
			x += sin_val;
		}
	}
	return x;
}

void do_main(int argc, char* argv[], int *status)
{
	int i, N, N2;
	double T1, T2;
	long delta_ms;
	double *M1, *M2, *M2_copy, *MERGED;
	int A = 540;
	unsigned int seed1, seed2;
	// double X;
	int iter = 10;

	N = atoi(argv[1]); /* N равен первому параметру командной строки */
	T1 = omp_get_wtime(); /* запомнить текущее время T1 */

	M1 = malloc(sizeof(double) * N);
	M2 = malloc(sizeof(double) * N / 2);
	M2_copy = malloc(sizeof(double) * N / 2);
	MERGED = malloc(sizeof(double) * N / 2);

	for (i = 0; i < iter; i++) /* 50 экспериментов */
	{	
		#if defined(_OPENMP)
		#pragma omp critical
		{
			*status = 100 * (i + 1) / iter;
		}
		#endif

		seed1 = i;
		seed2 = i;
		fill_array(M1, N, 1, A, &seed1);
		fill_array(M2, N / 2, A, 10 * A, &seed2);
		
		map_m1(M1, N);
		copy_arr(M2, N / 2, M2_copy);
		map_m2(M2, N / 2, M2_copy);

		apply_merge_func(M1, M2, N / 2);

		N2 = N / 2;

		#if defined(_OPENMP)
		#pragma omp parallel sections
		{
				#pragma omp section
				heapsort(M2, N2 / 2);
				#pragma omp section
				heapsort(M2 + (N2 / 2), (N2 + 1) / 2);
		}
		mergeArrays(MERGED, M2, M2 + (N2 / 2), N2 / 2, (N2 + 1) / 2);
		#else
		heapsort(M2, N2);
		#endif

		reduce(MERGED, N / 2);
		// printf("X = %f\n", X);
	}
	T2 = omp_get_wtime(); /* запомнить текущее время T2 */

	delta_ms = (T2 - T1) * 1000;
	printf("%d %ld\n", N, delta_ms); /* T2 - T1 */

	free(M1);
	free(M2);
	free(M2_copy);
	free(MERGED);
}

#if defined(_OPENMP)
void do_timer(int *status)
{
	int val;

	#pragma omp critical
	{
		val = *status;
	}

	while(val < 100) {
		#pragma omp critical
		{
			val = *status;
		}
		printf("Status = %d%%\n", val);
		sleep(1);
	}
}
#endif

int main(int argc, char* argv[])
{
	int status = 0;

	omp_set_nested(1);

	#if defined(_OPENMP)
	#pragma omp parallel sections shared(status)
	{
		#pragma omp section
		do_timer(&status);
		#pragma omp section
		do_main(argc, argv, &status);
	}
	#else
		do_main(argc, argv, &status);
	#endif

	return 0;
}
