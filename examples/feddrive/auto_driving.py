import drivetrainer
import client
from plato.servers import fedavg
from plato.algorithms import fedavg_personalized as per_fedavg
from datasource import DataSource

from timm.models import create_model

def main():
    # load the args used in vision encoder
    args, args_text = _parse_args()

    ad_model = create_model(
        args.model, # same as config.trainer.model_name
        pretrained=args.pretrained, # False
        drop_rate=args.drop, # 0.0
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp, # Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript, # convert model torchscript for inference, not give default
        checkpoint_path=args.initial_checkpoint,
        freeze_num=args.freeze_num,
    )

    # TODO
    # sampler function to get datasource for each client when simulating multiple clients with threads
    ad_datasource = DataSource(args)

    ad_trainer = drivetrainer.DriveTrainer(args)
    ad_algorithm = per_fedavg.Algorithm
    ad_client = client.Client(model=ad_model, datasource=ad_datasource, algorithm=ad_algorithm, trainer=ad_trainer)
    ad_server = fedavg.Server(model=ad_model, datasource=ad_datasource, algorithm=ad_algorithm, trainer=ad_trainer)

    ad_server.run(ad_client)

if __name__ == "__main__":
    main()