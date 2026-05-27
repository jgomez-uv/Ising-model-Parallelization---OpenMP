#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Prototypes.

int** initialize_spins_random(int L, int seed);

double calculate_energy(int** spins, int L, double J);
double calculate_energy_change(int** spins, int L, int i, int j, double J);
double calculate_total_magnetization(int** spins, int L);

void free_spins(int** spins, int L);
void print_config(int** spins, int L);
void save_config(int** spins, int L, const char* filename);

int metropolis_step(int** spins, int L, double J, double kT, double* mag);

void compute_correlation(int** spins, int L, int max_dist, double m, double* corr, int* counts);
double* average_correlation(int L, double J, double kT, int n_samples, int thermal_sweeps, int skip_sweeps, int seed);
double fit_correlation_length(double* corr, int max_dist, int fit_min, int fit_max);

int main(){

	// Exercise 1.
	
	int L = 32;
	double J = 1.0;
	double kT = 1.5;
	
	int** spins = initialize_spins_random(L,16331);
	
	save_config(spins, L, "initial_spins.csv");
	
	double energy = calculate_energy(spins, L, J);
	double magnetization = calculate_total_magnetization(spins, L) / (double)(L*L);
	printf("\nInitial Energy: %f\n",energy);
	printf("\nInitial Magnetization: %f\n",magnetization);
	
	unsigned long total_steps = 200 * L * L * (long)((float)L / log2(L));
	int measurement_interval = L * L;
	printf("Total Steps: %lu \n", total_steps);
	
	double* energy_history = (double*)malloc((total_steps/measurement_interval)*sizeof(double));
	double* magnetization_history = (double*)malloc((total_steps/measurement_interval)*sizeof(double));
	
	int step_index = 0;
	printf("Recording data... step, Energy, Magnetization\n");

	double dummy_magnetization = 0.0;
	srand(32502298);
	for (long step = 0; step < total_steps; step++){
		metropolis_step(spins,L,J,kT,&dummy_magnetization); // The last magnetization parameter is not used in this exercise.
		if (step % measurement_interval == 0){
			energy = calculate_energy(spins, L, J);
			magnetization = calculate_total_magnetization(spins, L)/(float)(L * L);
			
			energy_history[step_index] = energy;
			magnetization_history[step_index] = magnetization;
			step_index++;
			
			if (step_index % 100 == 0){
				printf("Step %ld of %lu, E = %.3f, M = %.3f\n", step, total_steps, energy, magnetization);
			}
		}
	}
	save_config(spins, L, "final_spins.csv");
	
	FILE* file = fopen("evolution.csv","w");
	for (int i = 0; i < step_index; i++){
		fprintf(file, "%d, %f, %f\n", i * measurement_interval, energy_history[i], magnetization_history[i]);
	}
	fclose(file);
	
	free(energy_history);
	free(magnetization_history);
	free_spins(spins,L);
	
	// Excercise 2

    L = 32;
    int thermal_sweeps = 1000;      // sweeps for equilibration
    int skip_sweeps = 100;          // sweeps between measurements
    int n_samples = 50;            // number of independent configurations
    int total_runs = 8;		    // number of independent runs per temperature

    FILE* out = fopen("xi_vs_T.csv", "w");
    
    double T_min = 2.0;
    double T_max = 3.0;
    double T_step = 0.05;
    
    int T_total = (int) ((T_max - T_min) / T_step) + 1;
    int i = 1;
    
    for (double T = T_min; T <= T_max; T += T_step) {
        double kT = T;
        printf("Computing T = %.2f, %d of %d\n", T, i, T_total);
        double xi = 0.0;
        
        for (int run = 0; run < total_runs; run++){
            printf("Run %d of %d\n",run+1,total_runs);
            double* corr = average_correlation(L, J, kT, n_samples, thermal_sweeps, skip_sweeps, 153*kT+923*run);
            // Choose fitting range: e.g. from R=1 to R=8 (avoid finite-size effects)
            int fit_min = 1;
            int fit_max = L/4;  // up to L/4 to stay away from periodic boundary artifacts
            xi += fit_correlation_length(corr, L/2, fit_min, fit_max);
            free(corr);
            }
        xi = xi / total_runs;
        i++;
        
        fprintf(out, "%f, %f\n", T, xi);
    }
    fclose(out);
}






int** initialize_spins_random(int L, int seed) {
	int** spins = (int**)malloc(L * sizeof(int*));
	
	for (int i = 0; i < L; i++){
		spins[i] = (int*)malloc(L * sizeof(int));
	}
	
	srand(seed);
	for (int i = 0; i < L; i++){
		for (int j = 0; j < L; j++){
			spins[i][j] = (rand() % 2) * 2 - 1;
		}
	}
	return spins;
}

double calculate_energy(int** spins, int L, double J){
	double energy = 0.0;
	for (int i = 0; i < L; i++){
		for (int j = 0; j < L; j++){
			int right_neighbor = spins[i][(j+1) % L];
			int down_neighbor = spins[(i+1) % L][j];
			
			energy += -J * spins[i][j] * (right_neighbor + down_neighbor);
		}
	}
	return energy;
}

double calculate_energy_change(int** spins, int L, int i, int j, double J){
	int current_spin = spins[i][j];
	
	int sum_neighbors = spins[(i - 1 + L) % L][j] + spins[(i + 1) % L][j] + spins[i][(j - 1 + L) % L] + spins[i][(j + 1) % L];
	
	return 2 * J * current_spin * sum_neighbors;
}

double calculate_total_magnetization(int** spins, int L){
	double magnetization = 0;    
	for (int i = 0; i < L; i++) {
		for (int j = 0; j < L; j++) {
			magnetization += spins[i][j];
		}
	}
	return magnetization;
}

void free_spins(int** spins, int L){
	for (int i = 0; i < L; i++){
		free(spins[i]);
	}
	free(spins);
}

void print_config(int** spins, int L){
	for (int i = 0; i < L; i++){
		for (int j = 0; j < L; j++){
			printf("%3d ", spins[i][j]);
		}
		printf("\n");
	}
}

void save_config(int** spins, int L, const char* filename){
	FILE* file = fopen(filename, "w");
	if (file == NULL){
		printf("Error opening file for saving config %s\n",filename);
		return;
	}
	fprintf(file, "# L = %d\n",L);
	for (int i = 0; i < L; i++){
		for (int j = 0; j < L; j++){
			fprintf(file,"%3d ", spins[i][j]);
		}
		fprintf(file,"\n");
	}
	fclose(file);
	printf("Config saved to %s\n",filename);
}

int metropolis_step(int** spins, int L, double J, double kT, double* mag){
	int i = rand() % L;
	int j = rand() % L;
	
	double delta_E = calculate_energy_change(spins, L, i, j, J);
	
	if (delta_E < 0 || (rand()/(double)(RAND_MAX)) < exp(-delta_E / kT)){
		int old = spins[i][j];
		spins[i][j] *= -1;
		*mag += -2.0 * (old);
		return 1;
	}
	else {
		return 0;
	}
}

void compute_correlation(int** spins, int L, int max_dist, double m, double* corr, int* counts) {
    for (int d = 0; d <= max_dist; d++){
    	corr[d] = 0.0;
    	counts[d] = 0;
    }

    // For each reference site (i,j)
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int s_ref = spins[i][j];
            // Horizontal direction: (i, j) to (i, j+dx) with dx = 1..max_dist
            for (int dx = 1; dx <= max_dist; dx++) {
                int j2 = (j + dx) % L;
                int s_other = spins[i][j2];
                corr[dx] += (double)(s_ref * s_other);
                counts[dx]++;
            }
            // Vertical direction: (i, j) to (i+dy, j) with dy = 1..max_dist
            for (int dy = 1; dy <= max_dist; dy++) {
                int i2 = (i + dy) % L;
                int s_other = spins[i2][j];
                corr[dy] += (double)(s_ref * s_other);
                counts[dy]++;
            }
        }
    }

    // Normalize and subtract the product of averages
    for (int d = 1; d <= max_dist; d++) {
        corr[d] = corr[d] / counts[d] - m * m;
    }
    corr[0] = 1.0 - m * m;   // R=0 gives variance of a single spin (<S^2> - <S>^2) = 1 - m^2
}

double* average_correlation(int L, double J, double kT, int n_samples, int thermal_sweeps, int skip_sweeps, int seed) {
    int max_dist = L/2;
    double* avg_corr = calloc(max_dist+1, sizeof(double));
    double* sample_corr = malloc((max_dist+1) * sizeof(double));
    int* counts = malloc((max_dist+1) * sizeof(int));

    // Initialize random spins
    int** spins = initialize_spins_random(L, seed);
    
    double total_mag = calculate_total_magnetization(spins, L);
    // Thermalize
    for (int step = 0; step < thermal_sweeps * L * L; step++) {
        metropolis_step(spins, L, J, kT, &total_mag);
    }

    // Accumulate correlation over independent samples
    for (int sample = 0; sample < n_samples; sample++) {
        // Skip "skip_sweeps" sweeps to decorrelate
        for (int s = 0; s < skip_sweeps * L * L; s++) {
            metropolis_step(spins, L, J, kT, &total_mag);
        }
        double m = total_mag / (L*L);
        // Compute correlation for this configuration
        compute_correlation(spins, L, max_dist, m, sample_corr, counts);
        for (int d = 0; d <= max_dist; d++) {
            avg_corr[d] += sample_corr[d];
        }
    }

    // Normalize by number of samples
    for (int d = 0; d <= max_dist; d++) {
        avg_corr[d] /= n_samples;
    }

    free_spins(spins, L);
    free(sample_corr);
    free(counts);
    return avg_corr;
}

double fit_correlation_length(double* corr, int max_dist, int fit_min, int fit_max) {
    int n = 0;
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    for (int r = fit_min; r <= fit_max; r++) {
        if (corr[r] > 1e-6) {       // avoid log(0) or very small values
            double y = log(corr[r]);
            sum_x += r;
            sum_y += y;
            sum_xy += r * y;
            sum_x2 += r * r;
            n++;
        }
    }
    if (n < 2) return 0.0;
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    return -1.0 / slope;   // ξ = -1/slope
}
