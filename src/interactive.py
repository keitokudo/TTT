import argparse
from trainer import Trainer

def main(args):
    trainer = Trainer(args, mode="predict")
    trainer.predict()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = Trainer.add_args(parser)
    args = parser.parse_args()
    main(args)
