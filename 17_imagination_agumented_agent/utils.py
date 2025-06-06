class MeanSummaryWriterWrapper():

    def __init__(self, writer, period):
        self.writer = writer
        self.period = period

        self.records = {}

    def __enter__(self):
        self.records = {}
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    def add_scalar(self, name, value, iter_idx):
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(value.detach().cpu())

        if len(self.records[name]) == self.period:
            self.writer.add_scalar(name, sum(self.records[name]) / len(self.records[name]), iter_idx)
            self.records[name] = []

    def add_histogram(self, name, value, iter_idx):
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(value.detach().cpu())