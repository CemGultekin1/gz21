from dataclasses import dataclass
from gz21.paths import SLURM_JOBS,SLURM_ECHO,EXT3,CUDA_SINGULARITY
import os

@dataclass
class SlurmJobHeaders:
    time:str = "1:00:00"
    array:str = "1"
    mem:str = "8GB"
    job_name:str = "none"
    output:str = SLURM_ECHO
    error:str = SLURM_ECHO
    nodes:str = 1
    ntasks_per_node:str = 1
    cpus_per_task :str = 1    
    def __post_init__(self,):
        self.output = os.path.join(self.output,self.job_name+"_%A_%a.out")
        self.error = os.path.join(self.error,self.job_name+"_%A_%a.err")
    def __repr__(self,):        
        st = "\n".join([f"#SBATCH --{key.replace('_','-')}={val}" for key,val in self.__dict__.items()])
        return "#!/bin/bash\n" + st
    @property
    def file_name(self,)->str:
        return self.job_name + '.s'
class SlurmJobBody:
    def __init__(self,read_only:bool = False,) -> None:
        rw = "ro" if read_only else "rw"
        self.environment = f"module purge\nsingularity exec --nv --overlay {EXT3}:{rw}\\\n\t {CUDA_SINGULARITY} /bin/bash -c \"\n"
        self.body = ["\t\tsource /ext3/env.sh;"]
        self.date = "echo \"$(date)\""
    def add_line(self,line:str):
        if line[-1]!=";":
            line += ";"
        self.body.append(line)
    def __repr__(self,):
        return self.date+"\n"+self.environment + "\n\t\t".join(self.body) +"\n\t\"\n" + self.date
def write_mlflow_slurm_job(
    entry_point:str,experiment_name:str,run_name:str,slurm_job_name:str,**kwargs
):
    sjh = SlurmJobHeaders(job_name = slurm_job_name,**kwargs)
    sjb = SlurmJobBody(read_only=True)
    sjb.add_line(f"mlflow run -e {entry_point} . --env-manager local --experiment-name {experiment_name} --run-name {run_name}")
    sb = "\n".join([str(sjh),str(sjb)])
    path = os.path.join(SLURM_JOBS,sjh.file_name)
    print(f'writing:\t {path}')
    with open(path,'w') as f:
        f.write(sb)
        
def main():
    write_mlflow_slurm_job("data-test","data","test","test",time = "10:00",mem = 4)
    write_mlflow_slurm_job("data","data","full","datagen",time = "24:00:00",mem = 24)
    
        
if __name__ == '__main__':
    main()
    