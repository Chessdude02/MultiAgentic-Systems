from sklearn.model_selection import GridSearchCV

class TuningAgent:
    def tune(self, model, X, y, param_grid):
        grid = GridSearchCV(model, param_grid, cv=3)
        grid.fit(X, y)
        return grid.best_estimator_
