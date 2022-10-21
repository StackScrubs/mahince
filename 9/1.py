from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from binary_bio_ai import BinaryBioAi, ModelParameters
import sys

def run_epochs(epochs, mp, verbose = False):
    avg_generations = 0
    avg_pop_size = 0
    time_measurements = []

    r = range(epochs) if verbose else tqdm(range(epochs))

    for _ in r:
        ai = BinaryBioAi(mp)

        done = False
        generation = 0
        epoch_start = time.perf_counter()

        if verbose:
            print(f"Generation: #{generation:07}: Target: {bin(mp.target)[2:].ljust(mp.n_genes, '0')} Population: {len(ai.population):03}")
        

        pop_sum = 0
        iters = 0
        while not done:
            ai.step()

            avg_fitness = ai.avg_fitness()
            best_fitness = ai.best_fitness()
            best_solution = ai.best_solution()
            
            if verbose:
                print(f"Generation: #{generation+1:07}: Target: {bin(mp.target)[2:].ljust(mp.n_genes, '0')} Best solution: {best_solution} Avg. fit: {avg_fitness:6.2f} Best fit: {best_fitness:04}, Population: {len(ai.population):03}")

            if best_fitness == 0:
                done = True
            
            generation += 1
            pop_sum += len(ai.population)
            iters += 1

        avg_generations += generation / epochs
        avg_pop_size += (pop_sum / iters) / epochs

        time_measurements.append(time.perf_counter() - epoch_start)

    return avg_generations, avg_pop_size, time_measurements

def run_single(num: int):
    run_epochs(1, ModelParameters(int(sys.argv[1]), 0.05, 10), True)

def run_all():
    EPOCHS = 1000
    N = 1
    P = 10
    avg_epoch_times = []
    bit_widths = []
    for i in range(N, N+P):
        print(f'\nRun {i} of {P}')
        bit_widths.append(i)
        mp = ModelParameters(i, 0.05, 10)
        avg_generations, avg_pop_size, time_measurements = run_epochs(EPOCHS, mp)
        run_time = sum(time_measurements)
        avg_epoch_time = run_time / EPOCHS
        avg_epoch_times.append(avg_epoch_time)
        print(f'UNIVERSE TERMINATED!\nBIT-WIDTH: {i}\nRun-time: {run_time:.2f}\nAvg. Epoch Time: {avg_epoch_time:.2f}s/epoch\nAvg. Generations {avg_generations:.2f}\nAvg. Pop. Size: {avg_pop_size:.2f}')
        
        x = [i for i in range(EPOCHS)]
        y = time_measurements
        plt.plot(x, y, color='green', label='Time pr. epoch')

        x = (0, EPOCHS)
        y = (avg_epoch_time, ) * 2
        plt.plot(x, y, color='red', label='Avg. time')

        plt.xlabel('Epochs')
        plt.ylabel('Time (s)')
        plt.title(f'Time measurements for bit-width {i}')
        plt.legend()
        plt.savefig(f'time_bit_width_{i}.png')
        plt.clf()

    plt.plot(bit_widths, avg_epoch_times, color='blue', label='Avg. epoch times pr. bit-width')

    plt.xlabel('Bit width')
    plt.ylabel('Time (s)')
    plt.title('Time measuresments')
    plt.legend()
    plt.savefig('time.png')
    plt.show()

def main():
    if len(sys.argv) >= 2:
        run_single(int(sys.argv[1]))
    else:
        run_all()

if __name__ == '__main__':
    main()
