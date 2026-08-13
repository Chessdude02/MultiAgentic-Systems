class SimpleSimulator:
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def trade_verbose(self, predictions, prices):
        cash = 1000
        position = 0
        history = []
        for i, (pred, price) in enumerate(zip(predictions, prices)):
            action = "HOLD"
            diff = (pred - price) / price
            if diff > self.threshold and cash >= price:
                position += 1
                cash -= price
                action = "BUY"
            elif diff < -self.threshold and position > 0:
                position -= 1
                cash += price
                action = "SELL"
            history.append({"Step": i, "Action": action, "Price": price, "Cash": cash, "Position": position})
        final_value = cash + position * prices[-1]
        return final_value, history