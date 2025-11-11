# models/gformer/Params.py
import argparse

def ParseArgs(cli_args=None):
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--ext', default=0.5, type=float)
    parser.add_argument('--gtw', default=0.1, type=float)
    parser.add_argument('--sub', default=0.1, type=float)
    parser.add_argument('--ctra', default=0.001, type=float)
    parser.add_argument('--b2', default=1, type=float)
    parser.add_argument('--anchor_set_num', default=32, type=int)
    parser.add_argument('--batch', default=4096, type=int)
    parser.add_argument('--seed', default=500, type=int)
    parser.add_argument('--tstBat', default=256, type=int)
    parser.add_argument('--reg', default=1e-4, type=float)
    parser.add_argument('--ssl_reg', default=1, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--decay', default=0.96, type=float)
    parser.add_argument('--save_path', default='tem')
    parser.add_argument('--latdim', default=32, type=int)
    parser.add_argument('--head', default=4, type=int)
    parser.add_argument('--gcn_layer', default=2, type=int)
    parser.add_argument('--gt_layer', default=1, type=int)
    parser.add_argument('--pnn_layer', default=1, type=int)
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--topk', default=20, type=int)
    parser.add_argument('--data', default='lastfm', type=str)
    parser.add_argument('--tstEpoch', default=3, type=int)
    parser.add_argument('--seedNum', default=9000, type=int)
    parser.add_argument('--maskDepth', default=2, type=int)
    parser.add_argument('--fixSteps', default=10, type=int)
    parser.add_argument('--keepRate', default=0.9, type=float)
    parser.add_argument('--keepRate2', default=0.7, type=float)
    parser.add_argument('--reRate', default=0.8, type=float)
    parser.add_argument('--addRate', default=0.01, type=float)
    parser.add_argument('--addNoise', default=0.0, type=float)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--eps', default=0.1, type=float)
    parser.add_argument('--approximate', dest='approximate', default=-1, type=int)

    # if cli_args is None, parse an empty list (so Jupyter args are ignored)
    if cli_args is None:
        args = parser.parse_args([])
    else:
        args = parser.parse_args(cli_args)

    # make sure user / item exist (default 0; wrapper will overwrite)
    args.user = 0
    args.item = 0
    return args

# create ONE global args used everywhere
args = ParseArgs()
