import sys
import os
from os.path import dirname, abspath
import importlib
import json

DeepETPickerHome = dirname(abspath(__file__))
DeepETPickerHome = os.path.split(DeepETPickerHome)[0]
sys.path.append(DeepETPickerHome)
sys.path.append(os.path.split(DeepETPickerHome)[0])
coords2labels = importlib.import_module(".utils.coords2labels", package=os.path.split(DeepETPickerHome)[1])
coord_gen = importlib.import_module(f".utils.coord_gen", package=os.path.split(DeepETPickerHome)[1])
norm = importlib.import_module(f".utils.normalization", package=os.path.split(DeepETPickerHome)[1])
option = importlib.import_module(f".options.option", package=os.path.split(DeepETPickerHome)[1])

if __name__ == "__main__":
    options = option.BaseOptions()
    args = options.gather_options()

    with open(args.pre_configs, 'r') as f:
        pre_config = json.loads(''.join(f.readlines()).lstrip('pre_config='))
     
    if "skip_coords" not in pre_config.keys():
        pre_config["skip_coords"] = False
    else:
        pre_config["skip_coords"] = bool(pre_config["skip_coords"])

    if "skip_labels" not in pre_config.keys():
        pre_config["skip_labels"] = False
    else:
        pre_config["skip_labels"] = bool(pre_config["skip_labels"])

    if "skip_ocp" not in pre_config.keys():
        pre_config["skip_ocp"] = False
    else:
        pre_config["skip_ocp"] = bool(pre_config["skip_ocp"])

    # initial coords
    if not pre_config["skip_coords"]: 
      coord_gen.coords_gen_show(args=(pre_config["coord_path"],
                                          pre_config["coord_format"],
                                          pre_config["base_path"],
                                          None,
                                          )
                                    )

    # normalization
    norm.norm_show(args=(pre_config["tomo_path"],
                         pre_config["tomo_format"],
                         pre_config["base_path"],
                         pre_config["norm_type"],
                         None,
                         )
                   )

    # generate labels based on coords
    if not pre_config["skip_labels"]:
      coords2labels.label_gen_show(args=(pre_config["base_path"],
                                          pre_config["coord_path"],
                                          pre_config["coord_format"],
                                          pre_config["tomo_path"],
                                          pre_config["tomo_format"],
                                          pre_config["num_cls"],
                                          pre_config["label_type"],
                                          pre_config["label_diameter"],
                                          None,
                                          )
                                    )

    # generate occupancy abased on coords
    if not pre_config["skip_ocp"]:
      coords2labels.label_gen_show(args=(pre_config["base_path"],
                                          pre_config["coord_path"],
                                          pre_config["coord_format"],
                                          pre_config["tomo_path"],
                                          pre_config["tomo_format"],
                                          pre_config["num_cls"],
                                          'data_ocp',
                                          pre_config["ocp_diameter"],
                                          None,
                                          )
                                    )
