import argparse
from train import train_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--symbol', default='AAPL')
    parser.add_argument('--period', default='6mo')
    args = parser.parse_args()

    if args.train:
        train_pipeline(symbol=args.symbol, period=args.period)
    else:
        print("Use --train to launch training")

if __name__ == '__main__':
    main()
