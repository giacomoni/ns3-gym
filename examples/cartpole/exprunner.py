import itertools
import os

if __name__ == "__main__":
    
    SEEDS =  [1,10]
    ENVS = ["ns3-v0"]
    WORKERS = [2,4,8,16,31]
    
    
    for params in itertools.product(ENVS, WORKERS, SEEDS):
        print(params) 
	#os.system(f"python3 simple_test.py {params[0]} {params[1]} {params[2]}")
