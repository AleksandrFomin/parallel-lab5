#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>

#define NTHREADS 4

#define pthread_parallel(arr1_param, arr_2_param, size_param, func_param) {					\
			int i;																			\
			void *retval;																	\
			pthread_t threads[NTHREADS];													\
			struct thread_arg *args = malloc(sizeof(struct thread_arg) * NTHREADS);			\
																							\
			for (i = 0; i < NTHREADS; i++) {												\
				args[i].arr = arr1_param;													\
				args[i].arr2 = arr_2_param;													\
				args[i].size = size_param;													\
				args[i].thread_id = i;														\
				pthread_create(&threads[i], NULL, func_param, args + i);					\
			}																				\
																							\
			for (i = 0; i < NTHREADS; i++)													\
				pthread_join(threads[i], &retval);											\
		}

struct thread_arg {
	double *arr;
	double *arr2;
	int    thread_id;
	int    size;
	double x;
	double y;
	unsigned int *seedp;
	int* status;
	int chunk;
};

pthread_mutex_t status_lock;
pthread_mutex_t shedule_dynamic_lock;

struct timeval timeval;
double omp_get_wtime()
{
	gettimeofday(&timeval, NULL);
	return (double)timeval.tv_sec + (double)timeval.tv_usec / 1000000;
}

void *fill_array_thread_func(void *arg)
{
	int i, j;
	struct thread_arg *arg_str = arg;

	int size = arg_str->size;
	double left = arg_str->x;
	double right = arg_str->y;
	double *arr = arg_str->arr;
	unsigned int *seedp = arg_str->seedp;
	int *global_i = arg_str->status;
	int chunk = arg_str->chunk;

	while(1) {
		pthread_mutex_lock(&shedule_dynamic_lock);
		i = *global_i;
		(*global_i) += chunk;
		pthread_mutex_unlock(&shedule_dynamic_lock);

		if (i >= size)
			break;

		for (j = i; j < (i + chunk) && j < size; j++, i++) {
			unsigned int seed_i = i + *seedp;
			arr[i] = rand_r(&seed_i) / (double)RAND_MAX * (right - left) + left;
		}
	}

	pthread_exit(NULL);
}

void fill_array(double *arr, int size, double left, double right, unsigned int *seedp)
{
	int i;
	int global_i = 0;
	void *retval;
	pthread_t threads[NTHREADS];
	struct thread_arg *args = malloc(sizeof(struct thread_arg) * NTHREADS);

	int chunk_size = 5;

	for (i = 0; i < NTHREADS; i++) {
		args[i].arr = arr;
		args[i].x = left;
		args[i].y = right;
		args[i].seedp = seedp;
		args[i].size = size;
		args[i].status = &global_i;
		args[i].chunk = chunk_size;
		pthread_create(&threads[i], NULL, fill_array_thread_func, args + i);
	}

	for (i = 0; i < NTHREADS; i++)
		pthread_join(threads[i], &retval);
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

void *map_m1_thread_func(void *arg)
{
	int i;
	struct thread_arg *arg_str = arg;

	int thread_id = arg_str->thread_id;
	int size = arg_str->size;
	double *arr = arg_str->arr;

	for (i = thread_id; i < size; i += NTHREADS) {
		arr[i] = tanh(arr[i]) - 1;
	}

	pthread_exit(NULL);
}

void map_m1(double *arr, int size)
{
	pthread_parallel(arr, NULL, size, map_m1_thread_func);
}

void *map_m2_thread_func(void *arg)
{
	int i;
	struct thread_arg *arg_str = arg;
	
	int thread_id = arg_str->thread_id;
	int size = arg_str->size;
	double *arr = arg_str->arr;
	double *arr_copy = arg_str->arr2;

	for (i = thread_id; i < size; i += NTHREADS) {
		double prev = 0;
		if (i > 0)
			prev = arr_copy[i - 1];
		arr[i] = sqrt(exp(1.0) * (arr_copy[i] + prev));
	}

	pthread_exit(NULL);
}

void map_m2(double *arr, int size, double *arr_copy)
{
	pthread_parallel(arr, arr_copy, size, map_m2_thread_func);
}

void *copy_arr_thread_func(void *arg)
{
	int i;
	struct thread_arg *arg_str = arg;
	
	int thread_id = arg_str->thread_id;
	int size = arg_str->size;
	double *src = arg_str->arr;
	double *dst = arg_str->arr2;

	for (i = thread_id; i < size; i += NTHREADS) {
		dst[i] = src[i];
	}

	pthread_exit(NULL);
}

void copy_arr(double *src, int len, double *dst)
{
	pthread_parallel(src, dst, len, copy_arr_thread_func);
}

void *apply_merge_thread_func(void *arg)
{
	int i;
	struct thread_arg *arg_str = arg;
	
	int thread_id = arg_str->thread_id;
	int size = arg_str->size;
	double *m1 = arg_str->arr;
	double *m2 = arg_str->arr2;

	for (i = thread_id; i < size; i += NTHREADS) {
		m2[i] = fabs(m1[i] - m2[i]);
	}

	pthread_exit(NULL);
}

void apply_merge_func(double *m1, double *m2, int m2_len)
{
	pthread_parallel(m1, m2, m2_len, apply_merge_thread_func);
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

void *reduce_thread_func(void *arg)
{
	int i;
	struct thread_arg *arg_str = arg;
	
	int thread_id = arg_str->thread_id;
	int size = arg_str->size;
	double *arr = arg_str->arr;
	double min_val = arg_str->x;
	double x = 0;

	for (i = thread_id; i < size; i += NTHREADS) {
		if ((int)(arr[i] / min_val) % 2 == 0) {
			double sin_val = sin(arr[i]);
			x += sin_val;
		}
	}

	arg_str->x = x;

	pthread_exit(NULL);
}

double reduce(double *arr, int len)
{
	int i;
	void *retval;
	double x = 0;
	pthread_t threads[NTHREADS];
	struct thread_arg *args = malloc(sizeof(struct thread_arg) * NTHREADS);

	double min_val = min_not_null(arr, len);

	for (i = 0; i < NTHREADS; i++) {
		args[i].arr = arr;
		args[i].x = min_val;
		args[i].size = len;
		args[i].thread_id = i;
		pthread_create(&threads[i], NULL, reduce_thread_func, args + i);
	}

	for (i = 0; i < NTHREADS; i++) {
		pthread_join(threads[i], &retval);
		x += args[i].x;
	}

	return x;
}

void *sort_thread_func(void *arg)
{
	struct thread_arg *arg_str = arg;
	
	int size = arg_str->size;
	double *arr = arg_str->arr;

	heapsort(arr, size);

	pthread_exit(NULL);
}

void *do_main(void *arg)
{
	int i, N2;
	void *retval;
	double T1, T2;
	long delta_ms;
	double *M1, *M2, *M2_copy, *MERGED;
	int A = 540;
	unsigned int seed1, seed2;
	// double X;
	int iter = 1;
	pthread_t threads[2];
	struct thread_arg *args = malloc(sizeof(struct thread_arg) * 2);

	struct thread_arg *arg_str = arg;
	
	int N = arg_str->y;
	int *status = arg_str->status;

	T1 = omp_get_wtime(); /* запомнить текущее время T1 */

	M1 = malloc(sizeof(double) * N);
	M2 = malloc(sizeof(double) * N / 2);
	M2_copy = malloc(sizeof(double) * N / 2);
	MERGED = malloc(sizeof(double) * N / 2);

	for (i = 0; i < iter; i++) /* 50 экспериментов */
	{	
		pthread_mutex_lock(&status_lock);
		*status = 100 * (i + 1) / iter;
		pthread_mutex_unlock(&status_lock);

		seed1 = i;
		seed2 = i;
		fill_array(M1, N, 1, A, &seed1);
		fill_array(M2, N / 2, A, 10 * A, &seed2);
		
		map_m1(M1, N);
		copy_arr(M2, N / 2, M2_copy);
		map_m2(M2, N / 2, M2_copy);

		apply_merge_func(M1, M2, N / 2);

		N2 = N / 2;

		args[0].arr = M2;
		args[0].size = N2 / 2;
		pthread_create(&threads[0], NULL, sort_thread_func, args);

		args[1].arr = M2 + (N2 / 2);
		args[1].size = (N2 + 1) / 2;
		pthread_create(&threads[1], NULL, sort_thread_func, args + 1);

		for (i = 0; i < 2; i++)
			pthread_join(threads[i], &retval);

		mergeArrays(MERGED, M2, M2 + (N2 / 2), N2 / 2, (N2 + 1) / 2);

		// print_array(MERGED, N / 2);

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

	pthread_exit(NULL);
}

void *do_timer(void *status)
{
	int val;

	pthread_mutex_lock(&status_lock);
	val = *((int *) status);
	pthread_mutex_unlock(&status_lock);

	while(val < 100) {
		pthread_mutex_lock(&status_lock);
		val = *((int *) status);
		pthread_mutex_unlock(&status_lock);
		printf("Status = %d%%\n", val);
		sleep(1);
	}

	pthread_exit(NULL);
}

int main(int argc, char* argv[])
{
	int status = 0;
	int i;
	void *retval;
	pthread_t threads[2];
	struct thread_arg *args = malloc(sizeof(struct thread_arg));
	int N = atoi(argv[1]);

	pthread_create(&threads[0], NULL, do_timer, &status);

	args->y = N;
	args->status = &status;
	pthread_create(&threads[1], NULL, do_main, args);

	for (i = 0; i < 2; i++)
		pthread_join(threads[i], &retval);

	return 0;
}
