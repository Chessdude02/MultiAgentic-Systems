import argparse
from train import DEFAULT_PERIOD, train_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--symbol', default='AAPL')
    parser.add_argument('--period', default=DEFAULT_PERIOD)
    parser.add_argument('--interval', default='1d')
    args = parser.parse_args()

    if args.train:
        train_pipeline(symbol=args.symbol, period=args.period, interval=args.interval)
    else:
        print("Use --train to launch training")

if __name__ == '__main__':
    main()
