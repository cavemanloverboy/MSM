

sim = lambda n: f"    --toml examples/go-streams/gaussian-overdensity-512-stream{n:05d}.toml \\\n"

file = open("run_streams.sh","w")
file.write("srun cargo run --release -- --verbose \\\n")

for stream in range(1,64+1):
    file.write(sim(stream))

file.close()


