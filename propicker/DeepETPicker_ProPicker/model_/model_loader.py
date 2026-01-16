from model_.residual_unet_att import ResidualUNet3D
from model_.conditioned_propicker import load_conditioned_propicker
    

def get_model(args):
    if args.network == 'ResUNet':
        model = ResidualUNet3D(f_maps=args.f_maps, out_channels=args.num_classes,
                               args=args, in_channels=args.in_channels, use_att=args.use_att,
                               use_paf=args.use_paf, use_uncert=args.use_uncert)
    
    if args.network == "ProPicker":
        model = load_conditioned_propicker(args)
    return model
    
