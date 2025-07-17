import os, json
import argparse
from common import loadConfig

def loadCfg(filename="config.json"):
    cfg = loadConfig(filename=filename)
    parser = argparse.ArgumentParser(description="Example Argument Parser")
    
    # Add command-line arguments
    parser.add_argument("--name", type=str, default=cfg.name, help="Name of the model (str)")
    parser.add_argument("--debug", action="store_true", default=False, help="debug")
    parser.add_argument("--use_pseudo", action="store_true", default=False, help="use_pseudo")
    args, opts = parser.parse_known_args()
    
    cfg.name = args.name
    cfg.debug = args.debug
    cfg.use_pseudo = args.use_pseudo
    
    for i in range(len(opts)):
        if opts[i].startswith("--"):
            k, v = opts[i][2::].strip(), opts[i+1].strip()
            dataType, v = v.split(":") if ":" in v else ("str", str(v))
            convert_func = {"int": lambda x:int(x), "float": lambda x:float(x), "str": lambda x:str(x), "bool": lambda x:x.lower()=="true", "":lambda x:x}
            complicate_func = {"listi":lambda x:list(map(int, x.split(","))), "listf": lambda x:list(map(float, x.split(","))), "lists":lambda x:list(map(str, x.split(",")))}
            func = convert_func | complicate_func
            v = func[dataType](v)
            setattr(cfg, k, v)
    
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.name)
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    cfg.env_dict = dict(os.environ)
    
    with open(f"{cfg.output_dir}/config.json", "w") as f:
        json.dump(vars(cfg), f, indent=4)
    
    return cfg