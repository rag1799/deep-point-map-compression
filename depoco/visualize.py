import depoco.utils.point_cloud_utils as pcu
import argparse
import ruamel.yaml as yaml
from depoco.trainer import DepocoNetTrainer
import torch
from ruamel.yaml import YAML 

if __name__ == "__main__":
    print('Hello')
    parser = argparse.ArgumentParser("./sample_net_trainer.py")
    parser.add_argument(
        '--config', '-cfg',
        type=str,
        required=False,
        default='config/depoco.yaml',
        help='configitecture yaml cfg file. See /config/config for example',
    )
    parser.add_argument(
        '--number', '-n',
        type=int,
        default=5,
        help='Number of maps to visualize',
    )
    FLAGS, unparsed = parser.parse_known_args()

    print('passed flags')
    #config = yaml.safe_load(open(FLAGS.config, 'r'))
    yaml = YAML(typ='safe', pure=True)
    config = yaml.load(open(FLAGS.config, 'r'))
    print('loaded yaml flags')
    trainer = DepocoNetTrainer(config)
    trainer.loadModel(best=True)
    print('initialized  trainer')
    for i, batch in enumerate(trainer.submaps.getOrderedTrainSet()):
        with torch.no_grad():
            print(batch)
            print("Before compression")
            pcu.visPointCloud(batch['points'].detach().cpu().numpy())
    
            points_est,features, point_attribute, nr_emb_points = trainer.encodeDecode(batch)
            print("After compression")
            print(
                f'nr embedding points: {nr_emb_points}, points out: {points_est.shape[0]}')
            pcu.visPointCloud(points_est.detach().cpu().numpy())
            
        if i+1 >= FLAGS.number:
            break
