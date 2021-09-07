# A-priori study of available parallelism

Studying the parallelism available inside a piece of code requires discussing multiple strategies for the parallelization of the serial implementation, and providing a theoretical assessment of the expected speed-up. The latter task is achieved by leveraging the Amdahl's law

$$
S = \frac{1}{1-p+\frac{p}{\sigma}}
$$

where $S$ is the overall obtainable speed-up, $p$ is the fraction of parallelizable code, and $\sigma$ is an enhancement factor (i.e., the speed-up) due to the parallelization of $p$. Our goal is to provide an assessment of the **maximum theoretical overall speedup** with respect to the number of threads available for parallelism.

### Estimating $p$

We estimate the parallelizable fraction of code through a profiler (i.e., a program that collects and arranges statistics on a piece of code). The default profiler available on Linux is `gprof` ( https://ftp.gnu.org/old-gnu/Manuals/gprof/ ). We eploy the latter to assess which functions are called by the driver program, and how much execution time each of them consumes, respectively. A [flat profile](https://ftp.gnu.org/old-gnu/Manuals/gprof/html_chapter/gprof_5.html#SEC11) of our program can be generated throught the following steps:

1. Compile and link the program with profiling enabled
   
   ```bash
   $ gcc -Wall -o test "/path/to/test.c" -pg
   ```

2. Execute the program to generate a profile data file
   
   ```
   $ ./test [<param>[, <param>]...]
   ```

3. Analyze the profile data
   
   ```bash
   $ gprof --flat-profile test > "/path/to/prof-file"
   ```

The  content of the output file is similar to the following

```textile
Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total
 time   seconds   seconds    calls   s/call   s/call  name
 56.14      7.08     7.08        1     7.08    12.44  mprod_rowcol
 42.40     12.44     5.35 1075838976     0.00     0.00  f_matrix_insert
  1.43     12.62     0.18        2     0.09     0.09  f_matrix_free
  0.16     12.64     0.02        2     0.01     0.01  f_matrix_randr
  0.08     12.65     0.01        1     0.01     0.01  f_matrix_calloc
  0.00     12.65     0.00        3     0.00     0.00  f_block_alloc
  0.00     12.65     0.00        3     0.00     0.00  f_matrix_alloc
  0.00     12.65     0.00        2     0.00     0.00  f_block_free
  0.00     12.65     0.00        2     0.00     0.01  f_matrix_init
```

in this case the functions `f_matrix_insert` and `mprod_rowcol` take up much the total execution time. The first function is a helper for inserting elements inside a matrix, while the second is the one implementing actual matrix multiplication. Since only the latter function can be parallelized, we assume $p = 56.14\%$.

### Estimating theoretical maximum speed-up

Once obtained the parallelizable fraction of code, we provide an estimate of the maximum achievable speed-up, using the Amdahl's law. In particular, since $\sigma \leq n$ always holds, $n$ being the number of threads, we compute the maximum speed-up as

$$
S(n) = \frac{1}{1-p+\frac{p}{n}}
$$

since $S \leq S(n)$.


