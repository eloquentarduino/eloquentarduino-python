class Voting:
    """
    Long-short term voting
    """
    def __init__(self, short, long):
        assert isinstance(short, tuple) and len(short) == 2, 'short MUST be a pair tuple'
        assert isinstance(long, tuple) and len(long) == 2, 'long MUST be a pair tuple'

        self.short_votes, self.short_quorum = short
        self.long_votes, self.long_quorum = long

        # allow for fractional quorum
        if self.short_quorum < 1:
            self.short_quorum = self.short_votes * self.short_quorum

        if self.long_quorum < 1:
            self.long_quorum = self.long_votes * self.long_quorum

        self.shorts = []
        self.longs = []

    def vote(self, vote):
        """
        Push vote to queue
        :param vote: int
        :return: int or None
        """
        self.shorts = self.shorts[-self.short_votes + 1:] + [vote]

        if self.shorts.count(vote) >= self.short_quorum:
            self.longs = self.longs[-self.long_votes + 1:] + [vote]

            if self.longs.count(vote) >= self.long_quorum:
                return vote

        return None
