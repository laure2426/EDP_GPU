/**************************************************************
This code is a part of a course on cuda taught by the author:
Lokman A. Abbas-Turki

Those who re-use this code should mention in their code 
the name of the author above.
Students : Laure MICHAUD - Tim HOUDAYE
***************************************************************/

#include <stdio.h>
#include <math.h>

#define EPS 0.0000001f
#define NTPB 256
#define NB 64
#define r 0.1f

typedef float MyTab[NB][NTPB];

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value 
// of the macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

/*************************************************************************/
/*                   Black-Sholes Formula                                */
/*************************************************************************/
/*One-Dimensional Normal Law. Cumulative distribution function. */
double NP(double x){
  const double p= 0.2316419;
  const double b1= 0.319381530;
  const double b2= -0.356563782;
  const double b3= 1.781477937;
  const double b4= -1.821255978;
  const double b5= 1.330274429;
  const double one_over_twopi= 0.39894228;  
  double t;

  if(x >= 0.0){
	t = 1.0 / ( 1.0 + p * x );
    return (1.0 - one_over_twopi * exp( -x * x / 2.0 ) * t * ( t *( t * 
		   ( t * ( t * b5 + b4 ) + b3 ) + b2 ) + b1 ));
  }else{/* x < 0 */
    t = 1.0 / ( 1.0 - p * x );
    return ( one_over_twopi * exp( -x * x / 2.0 ) * t * ( t *( t * ( t * 
		   ( t * b5 + b4 ) + b3 ) + b2 ) + b1 ));
  }
}


// Parallel cyclic reduction for implicit part
__device__ void PCR_d(float *sa, float *sd, float *sc, 
					  float *sy, int *sl, int n){
	// sa : Lower diagonal
	// sd : Mid diagonal
	// sc : Upper diagonal
	// sy : Vector of value to find
	// sl : Vector of permutation
	// n  : Number of threads per block

	int i, lL, d, tL, tR;
	float aL, dL, cL, yL;
	float aLp, dLp, cLp, yLp;

	d = (n/2+(n%2))*(threadIdx.x%2) + (int)threadIdx.x/2;

	tL = threadIdx.x - 1;
	if (tL < 0) tL = 0;
	tR = threadIdx.x + 1;
	if (tR >= n) tR = 0;
	
	for(i=0; i<(int)(logf((float)n)/logf(2.0f))+1; i++){
		lL = (int)sl[threadIdx.x];

		aL = sa[threadIdx.x];
		dL = sd[threadIdx.x];
		cL = sc[threadIdx.x];
		yL = sy[threadIdx.x];

		dLp = sd[tL];
		cLp = sc[tL];

		if(fabsf(aL) > EPS){
			aLp = sa[tL];
			yLp = sy[tL];
			dL -= aL*cL/dLp;
			yL -= aL*yLp/dLp;
			aL = -aL*aLp/dLp;
			cL = -cLp*cL/dLp;
		}
		
		cLp = sc[tR];
		if(fabsf(cLp) > EPS){
			aLp = sa[tR];
			dLp = sd[tR];
			yLp = sy[tR];
			dL -= cLp*aLp/dLp;
			yL -= cLp*yLp/dLp;
		}
		__syncthreads();

		if (i < (int)(logf((float)n)/logf(2.0f))){
			sa[d] = aL;
			sd[d] = dL;
			sc[d] = cL;
			sy[d] = yL;
			sl[d] = (int)lL;	
			__syncthreads();
		}
	}

	sy[(int)sl[threadIdx.x]] = yL / dL;
}

/////////////////////////////////////////////////////////////////////////////
//                        Explicit Scheme Methods                          //
/////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////
//    A bad solution that makes a lot of accesses to the global memory     //
/////////////////////////////////////////////////////////////////////////////
__global__ void PDE_diff_k1 (float dt, float dx, float dsig, float pmin, 
							 float pmax, float sigmin, MyTab *pt_GPU){
	// sigma
	float sigma = (sigmin + blockIdx.x*dsig);

	// mu
	float mu = r - sigma*sigma/2.0f;
	
	// Variables for the computation of p_u, p_m, p_d
	float p_u = sigma*sigma*dt / (2.0f*dx*dx) + mu*dt/(2.0f*dx);
	float p_d = sigma*sigma*dt / (2.0f*dx*dx) - mu*dt/(2.0f*dx);
	float p_m = 1.0f - sigma*sigma*dt / (dx*dx);

	// Shared float to store the price of the option
	__shared__ float result[NTPB];
	

	// Compute the price of the option given the data from the pt_GPU 
	// Lower bound of the dx space
	if (threadIdx.x == 0){
		result[threadIdx.x] = pmin;
	}
	// Upper bound of the dx space
	else if(threadIdx.x == (NTPB - 1)){
		result[threadIdx.x] = pmax;
	}
	else{
		result[threadIdx.x] = p_u * pt_GPU[0][blockIdx.x][threadIdx.x + 1] 
							+ p_m * pt_GPU[0][blockIdx.x][threadIdx.x] 
							+ p_d * pt_GPU[0][blockIdx.x][threadIdx.x - 1];
	}                                                                        
  
	// Wait for all the threads to compute
	__syncthreads();

	// Rewrite on the pt_GPU
	pt_GPU[0][blockIdx.x][threadIdx.x] = result[threadIdx.x];

}


/////////////////////////////////////////////////////////////////////////////
//              A good solution with dynamic shared memory                 //
/////////////////////////////////////////////////////////////////////////////
__global__ void PDE_diff_k2 (int N, float dt, float dx, float dsig, float pmin, 
							float pmax, float sigmin, MyTab *pt_GPU){
	// sigma
	float sigma = (sigmin + blockIdx.x*dsig);

	// mu
	float mu = r - sigma*sigma/2.0f;
	
	// Variables for the computation of p_u, p_m, p_d
	float p_u = sigma*sigma*dt / (2.0f*dx*dx) + mu*dt/(2.0f*dx);
	float p_d = sigma*sigma*dt / (2.0f*dx*dx) - mu*dt/(2.0f*dx);
	float p_m = 1.0f - sigma*sigma*dt / (dx*dx);
	
	// Variable shared memory
	extern __shared__ float result_block[];

	// Compute the price of the option given the data from the pt_GPU 
	if (threadIdx.x == 0){
		result_block[threadIdx.x] = pmin;
	}
	// Upper bound of the dx space
	else if(threadIdx.x == (NTPB - 1)){
		result_block[threadIdx.x] = pmax;
	}
	else{
		result_block[threadIdx.x] = p_u * pt_GPU[0][blockIdx.x][threadIdx.x + 1] 
								  + p_m * pt_GPU[0][blockIdx.x][threadIdx.x] 
								  + p_d * pt_GPU[0][blockIdx.x][threadIdx.x - 1];
	}

	// Wait for all the threads to end before starting the loop on time 
	__syncthreads();

	// counter for the date. i.e Start at 1 because we already compute the first time
	int i_date = 1;

	// Loop on date
	while (i_date < N){

		// Shared float to store the price of the option at each time i_date
		__shared__ float result_thread[NTPB];

		// Compute the price of the option given the data from the result_block 
		// Lower bound of the dx space
		if (threadIdx.x == 0){
			result_thread[threadIdx.x] = pmin;
		}
		// Upper bound of the dx space
		else if(threadIdx.x == (NTPB - 1)){
			result_thread[threadIdx.x] = pmax;
		}
		else{
			result_thread[threadIdx.x] = p_u * result_block[threadIdx.x + 1] 
									   + p_m * result_block[threadIdx.x] 
									   + p_d * result_block[threadIdx.x - 1];
		}

		// Wait for all the thread to compute before writting the new data on the dynamic shared variable: result_block
		__syncthreads();
		result_block[threadIdx.x] = result_thread[threadIdx.x];
		__syncthreads();
		i_date += 1;
	}

	// Wait for all the threads to compute
	__syncthreads();

	// Rewrite on the pt_GPU
	pt_GPU[0][blockIdx.x][threadIdx.x] = result_block[threadIdx.x];
}


/////////////////////////////////////////////////////////////////////////////
//                        Implicit Scheme Methods                          //
/////////////////////////////////////////////////////////////////////////////

__global__ void PDE_diff_k3 (int N, float dt, float dx, float dsig, float pmin, 
							float pmax, float sigmin, MyTab *pt_GPU){
	// sigma
	float sigma = (sigmin + blockIdx.x * dsig);

	// mu
	float mu = r - sigma*sigma/2.0f;

	// Variable for the dynamic shared memory
	extern __shared__ float sy[];

	// Create shared vector for the 3 diagonals of the matrix of q_u, q_m, q_d
  	__shared__ float sa[NTPB];
	__shared__ float sd[NTPB];
	__shared__ float sc[NTPB];
	__shared__ int sl[NTPB];

	// Create variables q_u, q_m, q_d which are constant
	float q_u = -1.0f * sigma*sigma*dt / (2.0f*dx*dx) - mu*dt/(2.0f*dx);
	float q_m =  1.0f + sigma*sigma*dt / (dx*dx);
	float q_d = -1.0f * sigma*sigma*dt / (2.0f*dx*dx) + mu*dt/(2.0f*dx);

	// Fetch the value in the dynamic shared variable
	sy[threadIdx.x] = pt_GPU[0][blockIdx.x][threadIdx.x];
	
	// Conditions on bounds of dx space
	if (threadIdx.x == 0){
		sy[threadIdx.x] = pmin;
	}
	// Upper bound of the dx space
	else if(threadIdx.x == (NTPB - 1)){
		sy[threadIdx.x] = pmax;
	}
	__syncthreads();

	// counter for the date
	int i_date = 0;

	while(i_date < N){
		// Compute the three diagonals
		// Lower diagonal
		sa[threadIdx.x] = (threadIdx.x == 0)*(0.0f) \
						+ (threadIdx.x != 0)*(q_d);
		// Mid diagonal
		sd[threadIdx.x] = q_m;
		// Upper diagonal
		sc[threadIdx.x] = (threadIdx.x == (NTPB - 1))*(0.0f) \
						+ (threadIdx.x != (NTPB - 1))*(q_u);

		// In sl stack the threadIdx.x number
		sl[threadIdx.x] = threadIdx.x;
		__syncthreads();
		
		// Compute the price of the put option using the PCR methods
		PCR_d(sa, sd, sc, sy, sl, NTPB);
		__syncthreads();

		// Conditions on bounds of dx space
		if (threadIdx.x == 0){
			sy[threadIdx.x] = pmin;
		}
		// Upper bound of the dx space
		else if(threadIdx.x == (NTPB - 1)){
			sy[threadIdx.x] = pmax;
		}

		// Wait for all the thread to finish and go to the next date
		__syncthreads();
		i_date += 1;
  	}
	// Once finish put back the result to the pt_GPU variable
  	__syncthreads();
  	pt_GPU[0][blockIdx.x][threadIdx.x] = sy[threadIdx.x];
}


/////////////////////////////////////////////////////////////////////////////
//                        Crank Nicholsons Methods                         //
/////////////////////////////////////////////////////////////////////////////

__global__ void PDE_diff_k4 (int N, float dt, float dx, float dsig, float pmin, 
							float pmax, float sigmin, MyTab *pt_GPU){
	// sigma
	float sigma = (sigmin + blockIdx.x*dsig);

	// mu
	float mu = r - sigma*sigma/2.0f;

	// Variables for the computation of p_u, p_m, p_d
	float p_u = sigma*sigma*dt / (2.0f*dx*dx) + mu*dt/(2.0f*dx);
	float p_d = sigma*sigma*dt / (2.0f*dx*dx) - mu*dt/(2.0f*dx);
	float p_m = 1.0f - sigma*sigma*dt / (dx*dx);

	// Create variables q_u, q_m, q_d
	float q_u = -1.0f * sigma*sigma*dt / (2.0f*dx*dx) - mu*dt/(2.0f*dx);
	float q_m = 1.0f + sigma*sigma*dt/(dx*dx);
	float q_d = -1.0f * sigma*sigma*dt / (2.0f*dx*dx) + mu*dt/(2.0f*dx);

	// Variable for the dynamic shared memory
	extern __shared__ float sy[];

	// Create shared vector for the 3 diagonals of the matrix
	__shared__ float sa[NTPB];
	__shared__ float sd[NTPB];
	__shared__ float sc[NTPB];
	__shared__ int sl[NTPB];

	// Compute diagonals using q_d, q_m, q_u
	// Lower diagonal
	sa[threadIdx.x] = (threadIdx.x == 0)*(0.0f) \
					+ (threadIdx.x != 0)*(q_d);
	// Mid diagonal
	sd[threadIdx.x] = q_m;
	// Upper diagonal
	sc[threadIdx.x] = (threadIdx.x == (NTPB - 1))*(0.0f) \
					+ (threadIdx.x != (NTPB - 1))*(q_u);

	// Compute the explicit part of Crank-Nicolson scheme
	// Lower bound of dx space
	if (threadIdx.x == 0){
		sy[threadIdx.x] = pmin;
	}
	// Upper bound of dx space
	else if(threadIdx.x == (NTPB - 1)){
		sy[threadIdx.x] = pmax;
	}
	else{
		sy[threadIdx.x] = p_u * pt_GPU[0][blockIdx.x][threadIdx.x + 1] 
						+ p_m * pt_GPU[0][blockIdx.x][threadIdx.x] 
						+ p_d * pt_GPU[0][blockIdx.x][threadIdx.x - 1];
	}

	// In sl stack the threadIdx.x number
	sl[threadIdx.x] = threadIdx.x;
	__syncthreads();

	// Compute the price of the option using the PCR_d algorithm
	PCR_d(sa, sd, sc, sy, sl, NTPB);

	// Conditions on bounds of dx space
	if (threadIdx.x == 0){
		sy[threadIdx.x] = pmin;
	}
	else if(threadIdx.x == (NTPB - 1)){
		sy[threadIdx.x] = pmax;
	}

	// counter for the date. i.e Start at 1 because we already compute the first time
	__syncthreads();
	int i_date = 1;

	while(i_date < N){

		// Shared float to store the price of the option at each time i_date
		__shared__ float result_thread[NTPB];

		// Compute diagonals using q_d, q_m, q_u
		// Lower diagonal
		sa[threadIdx.x] = (threadIdx.x == 0)*(0.0f) \
						+ (threadIdx.x != 0)*(q_d);
		// Mid diagonal
		sd[threadIdx.x] = q_m;
		// Upper diagonal
		sc[threadIdx.x] = (threadIdx.x == (NTPB - 1))*(0.0f) \
						+ (threadIdx.x != (NTPB - 1))*(q_u);

		// Compute the explicit part of Crank-Nicolson scheme
		// Lower bound of dx space
		if (threadIdx.x == 0){
			result_thread[threadIdx.x] = pmin;
		}
		// Upper bound of dx space
		else if(threadIdx.x == (NTPB - 1)){
			result_thread[threadIdx.x] = pmax;
		}
		else{
			result_thread[threadIdx.x] = p_u * sy[threadIdx.x + 1] 
									   + p_m * sy[threadIdx.x] 
									   + p_d * sy[threadIdx.x - 1];
		}
		
		// In sl stack the threadIdx.x number
		sl[threadIdx.x] = threadIdx.x;
		__syncthreads();

		// Write the new values of u_{i+1} in sy
		sy[threadIdx.x] = result_thread[threadIdx.x];
		__syncthreads();

		// Compute the price of the option using the PCR_d algorithm
		PCR_d(sa, sd, sc, sy, sl, NTPB);

		// Conditions on bounds of dx space
		if (threadIdx.x == 0){
			sy[threadIdx.x] = pmin;
		}
		else if(threadIdx.x == (NTPB - 1)){
			sy[threadIdx.x] = pmax;
		}

		// Wait for all the thread to finish
		__syncthreads();
		i_date += 1;
	}
	// Once finish put back the result to the pt_GPU variable
	__syncthreads();
	pt_GPU[0][blockIdx.x][threadIdx.x] = sy[threadIdx.x];
}



/////////////////////////////////////////////////////////////////////////////
//                        Find the optimal x_0 and sigma_0                 //
/////////////////////////////////////////////////////////////////////////////
__global__ void Optimal_k1 (float u_0, float dx, float dsig, 
							float xmin, float sigmin, MyTab *pt_GPU){

    // Shared dynamic variable to find the minimum value for the x dimension
	extern __shared__ float distance_values[];

	// Shared static variable to stack x values
	__shared__ float dx_values[NTPB];
					
	// Get the absolute distance between u_{i,j} and u_0 and the price 
	distance_values[threadIdx.x] = abs(pt_GPU[0][blockIdx.x][threadIdx.x] - u_0);
	dx_values[threadIdx.x] = (xmin + threadIdx.x * dx);
	
	// Wait for all the threads
	__syncthreads();
	
	// Loop on threads within a block to find the minimal distance and its corresponding price
	int i_thread = blockDim.x/2;
	while (i_thread != 0){
		if (threadIdx.x < i_thread){
			// If the value in threadIdx.x + i_thread is lower, fetch this value and the price associated 
			if (distance_values[threadIdx.x] > distance_values[threadIdx.x + i_thread]){
				distance_values[threadIdx.x] = distance_values[threadIdx.x + i_thread];	
				dx_values[threadIdx.x] = dx_values[threadIdx.x + i_thread];
			}
        		
		}
		__syncthreads();
		i_thread /= 2;
	}
	__syncthreads();

	// Fetch the minimum values between blocks using pt_GPU

	if (threadIdx.x == 0){
    	// Add values in pt_GPU: u_min, x_min, sigma
		pt_GPU[0][blockIdx.x][0] = distance_values[threadIdx.x];
		pt_GPU[0][blockIdx.x][1] = dx_values[threadIdx.x];
		pt_GPU[0][blockIdx.x][2] = (sigmin + blockIdx.x * dsig);
    	__syncthreads();

    	// Loop on blocks
		int i_block = NB/2;
		while (i_block != 0){
			if (blockIdx.x < i_block){
				if(pt_GPU[0][blockIdx.x][0] > pt_GPU[0][blockIdx.x + i_block][0]){
					pt_GPU[0][blockIdx.x][0] = pt_GPU[0][blockIdx.x + i_block][0];
					pt_GPU[0][blockIdx.x][1] = pt_GPU[0][blockIdx.x + i_block][1];
					pt_GPU[0][blockIdx.x][2] = pt_GPU[0][blockIdx.x + i_block][2];
				}
			}
			__syncthreads();
			i_block /= 2;
		}
	}
}

// Wrapper 
void PDE_diff (float dt, float dx, float dsig, float pmin, float pmax, 
			   float sigmin, int N, float xmin, MyTab* CPUTab){

	float TimeExec;						// GPU timer instructions
	cudaEvent_t start, stop;			// GPU timer instructions
	testCUDA(cudaEventCreate(&start));  // GPU timer instructions
	testCUDA(cudaEventCreate(&stop));	// GPU timer instructions
	testCUDA(cudaEventRecord(start,0));	// GPU timer instructions

	MyTab *GPUTab;

	testCUDA(cudaMalloc(&GPUTab, sizeof(MyTab)));
	testCUDA(cudaMemcpy(GPUTab, CPUTab, sizeof(MyTab), cudaMemcpyHostToDevice));
	
	// Question 1.
	// Accessing 2*N times to the global memory
	for(int i=0; i<N; i++){
	   PDE_diff_k1<<<NB,NTPB>>>(dt, dx, dsig, pmin, pmax, sigmin, GPUTab);
	}
  	
	/*
	// Question 2.
	PDE_diff_k2<<<NB, NTPB, NTPB*sizeof(float)>>>(N, dt, dx, dsig, pmin, pmax, sigmin, GPUTab);
	*/

	/*
	// Question 4.
	PDE_diff_k3<<<NB, NTPB, NTPB*sizeof(float)>>>(N, dt, dx, dsig, pmin, pmax, sigmin, GPUTab);
	*/

	/*
	// Question 5.
	PDE_diff_k4<<<NB, NTPB, NTPB*sizeof(float)>>>(N, dt, dx, dsig, pmin, pmax, sigmin, GPUTab);
	*/

	/*
	// Question 6.
  Optimal_k1<<<NB, NTPB, NTPB*sizeof(float)>>>(4.146348, dx, dsig, xmin, sigmin, GPUTab);
  */

	testCUDA(cudaMemcpy(CPUTab, GPUTab, sizeof(MyTab), cudaMemcpyDeviceToHost));

	testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
	testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	testCUDA(cudaEventElapsedTime(&TimeExec,	// GPU timer instructions
			 start, stop));							          // GPU timer instructions
	testCUDA(cudaEventDestroy(start));				// GPU timer instructions
	testCUDA(cudaEventDestroy(stop));				  // GPU timer instructions

	printf("GPU time execution for PDE diffusion: %f ms\n", TimeExec);

	testCUDA(cudaFree(GPUTab));	
}

///////////////////////////////////////////////////////////////////////////
// main function for a put option f(x) = max(0,K-x)
///////////////////////////////////////////////////////////////////////////
int main(void){

	float S0 = 100.0f;
	float K = 100.0f;
	float T = 1.0f;
	int N = 10000;
	float dt = (float)T/N;
	float xmin = log(0.5f*S0);
	float xmax = log(2.0f*S0);
	float dx = (xmax-xmin)/NTPB;
	float pmin = 0.5f*K;
	float pmax = 0.0f;
	float sigmin = 0.1f;
	float sigmax = 0.5f;
	float dsig = (sigmax-sigmin)/NB;
	

	MyTab *pt_CPU;

	testCUDA(cudaHostAlloc(&pt_CPU, sizeof(MyTab), cudaHostAllocDefault));

	// Terminal Condition
	for(int i=0; i<NB; i++){
	   for(int j=0; j<NTPB; j++){
	      pt_CPU[0][i][j] = max(0.0, K-exp(xmin + dx*j));	
	   }	
	}

	PDE_diff(dt, dx, dsig, pmin, pmax, sigmin, N, xmin, pt_CPU);

    // S0 = 100 , sigma = 0.2
    printf("%f",pt_CPU[0][16][128]);
	printf(" %f, compare with %f\n",exp(-r*T)*pt_CPU[0][16][128],
		   K*(exp(-r*T)*NP(-(r-0.5*0.2*0.2)*sqrt(T)/0.2)-
		   NP(-(r+0.5*0.2*0.2)*sqrt(T)/0.2)));
	// S0 = 100 , sigma = 0.3
	printf(" %f, compare with %f\n",exp(-r*T)*pt_CPU[0][32][128],
		   K*(exp(-r*T)*NP(-(r-0.5*0.3*0.3)*sqrt(T)/0.3)-
		   NP(-(r+0.5*0.3*0.3)*sqrt(T)/0.3)));
	// S0 = 141.4214 , sigma = 0.3
	printf(" %f, compare with %f\n",exp(-r*T)*pt_CPU[0][32][192],
		   K*exp(-r*T)*NP(-(log(141.4214/K)+(r-0.5*0.3*0.3)*T)/(0.3*sqrt(T)))-
		   141.4214*NP(-(log(141.4214/K)+(r+0.5*0.3*0.3)*T)/(0.3*sqrt(T))));
	
	// Print the optimal couple (x, sigma) for a given u
	/*
	printf("Optimal (x,sigma) couple given a u");
	printf(" S0: %f, sigma: %f, compare with S0: 100 and sigma: 0.2\n", exp(pt_CPU[0][0][1]), pt_CPU[0][0][2]);
	*/

	testCUDA(cudaFreeHost(pt_CPU));	
	return 0;
}
