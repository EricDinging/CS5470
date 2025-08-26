# CS5470
Welcom to CS5470! This repo contains the three assignments that you are going to complete in this course. We are going to use [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/) to run ML workloads.

## Working on Perlmutter
[Getting Started at NERSC](https://docs.nersc.gov/getting-started/) contains helpful overview of the cluster. Please make sure to read the documentation.

When you first connect to Perlmutter, you will be on the login node! This is the place where you install environments, libraries, ane compile your workloads. This is NOT the place to execute jobs, such as training, fine-tuning, or serving workloads. You should only run computation on compute nodes through the interactive queue or (better) batch [script](https://my.nersc.gov/script_generator.php).

If you have any questions about the account process or system, please feel free to contact the NERSC help desk via their [ticket system](https://nersc.servicenowservices.com/sp).

## Prepare conda environment
```
# on login node
module load conda
```





