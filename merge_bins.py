
distinct_probs = {}

def max_sampling_prob(bins):
    #Num of (true) bins after merging
    true_bins = [bin for bin in bins if bin != []]

    #Sampling prob for each value
    max_prob = 0
    for bin in true_bins:
        if sum(bin)!=0:                
            tmp_prob = float(1)/(float(sum(bin)*len(true_bins)))
            max_prob = max(tmp_prob,max_prob)
    return max_prob


def checkConstraint(bins, threshold):
    for bin in bins:
        tmp = sum(bin)
        if tmp < threshold and len(bin)!=0:
            return False
    return True


def binCounts(bins, counts, i, threshold):
   
    if i == len(counts):
        if checkConstraint(bins, threshold):
            prob = max_sampling_prob(bins)
            
            if prob not in distinct_probs:
                distinct_probs[prob] = bins
                print(prob,":",bins)
        return
    
    for bin in bins:
        bin.append(counts[i])
        binCounts(bins, counts, i+1, threshold)
        bin.pop()

if __name__ == "__main__":
    num_counts = 8
    num_bins = 8
    k = 4

    counts = list(range(num_counts))
    bins = [[] for i in range(num_bins)]

    binCounts(bins, counts, 0, k);