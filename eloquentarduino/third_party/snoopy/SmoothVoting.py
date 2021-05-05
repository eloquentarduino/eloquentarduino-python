class SmoothVoting:
    def __init__(self, decay=0.7, vote_thresh=0.2, var_thresh=0.2):
        self.k = 0
        self.mean = 0
        self.var = 0
        self.decay = decay
        self.vote_thresh = vote_thresh
        self.var_thresh = var_thresh

    def vote(self, vote):
        self.push(vote)

        if self.k > 1 and abs(vote - self.mean) < self.vote_thresh and self.var < self.var_thresh:
            return vote

        return None

    def push(self, vote):
        self.k += 1
        mean = self.decay * self.mean + (1 - self.decay) * vote

        if self.k > 1:
            var = self.var + (vote - self.mean) * (vote - mean)
            self.var = var / self.k

        self.mean = mean

        return self
