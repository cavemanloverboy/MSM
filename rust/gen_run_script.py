import numpy as np

sim = lambda n: f"    --toml examples/go-streams/gaussian-overdensity-512-stream{n:05d}.toml \\\n"

file = open("run_streams.sh","w")
file.write("srun cargo run --release -- --verbose \\\n")

for stream in np.arange(1,64+1) + 64:
    file.write(sim(stream))

file.close()


