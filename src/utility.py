

#  Helper function to create weighted rvs
def weighted_probs(outcomes, probabilities, size):
   	temp = np.add.accumulate(probabilities)
		 return outcomes[np.digitize(random_sample(size), temp)]


#  methods for byte seq randomness from np. Useful for creating/filling in dataframe
#  Return probabilities between 0 and 1
def rand_prob(self, n):
		return np.random.random((n)).tolist()

#Return random values between low and high
def rand_range(self, lo, high, n):
		return np.random.randint(lo,high,n).tolist()