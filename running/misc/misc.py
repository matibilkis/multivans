class FakeArgs:
    def __init__(self,args):
        self.problem = args["problem"]
        self.params = args["params"]
        self.nrun = args["nrun"]
        self.shots = args["shots"]
        self.epochs = args["epochs"]
        self.n_qubits = args["n_qubits"]
        self.vans_its = args["vans_its"]
        self.itraj = args["itraj"]
        self.noisy = args["noisy"]
        self.noise_strength = args["noise_strength"]
def convert_shorts(x):
    if x==0:
        return None
    else:
        return x
