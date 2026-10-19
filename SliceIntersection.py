"""Keep statements covered by every failing test, retaining all test rows."""
from ProcessedData import ProcessedData


class SliceIntersection(ProcessedData):
    def process(self):
        if not self.feature_df.index.equals(self.label_df.index):
            raise ValueError("Coverage and test labels must have matching row indices")
        failing = self.label_df["error"] == 1
        if not failing.any():
            raise ValueError("Execution slicing requires at least one failing test")
        selected = (self.feature_df.loc[failing] > 0).all(axis=0)
        if not selected.any():
            raise ValueError("Failing-test coverage intersection is empty; no statement is covered by every failing test")
        excluded = self.feature_df.columns[~selected].tolist()
        self.rest_columns = list(dict.fromkeys(self.rest_columns + excluded))
        before = self.feature_df.shape[1]
        self.feature_df = self.feature_df.loc[:, selected].copy()
        self.data_df = self.feature_df.join(self.label_df)
        print("Execution slice (intersection): {}/{} statements; {} failing tests; all {} test rows retained".format(
            self.feature_df.shape[1], before, int(failing.sum()), len(self.label_df)))
