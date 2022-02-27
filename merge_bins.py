import random

distinct_probs = {}

# Get worst-case sampling probability
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


def get_distances(bins):
    distances = {}
    counts = 0
    for bin in bins:
        if len(bin)==0:
            continue
        elif len(bin)==1: 
            distances[bin[0]] = [None,None] 
        elif len(bin)==2:
            srtdb = sorted(bin)
            distances[srtdb[0]] = [None, srtdb[1]-srtdb[0]]
            distances[srtdb[1]] = [srtdb[1]-srtdb[0], None]
        else:
            srtdb = sorted(bin)            
            for i,v in enumerate(srtdb):

                if i==0:
                    distances[v] = [None, srtdb[1]-srtdb[0]]
                elif i==len(srtdb)-1:
                    distances[v] = [srtdb[i]-srtdb[i-1], None]
                else:
                    distances[v] = [srtdb[i]-srtdb[i-1], srtdb[i+1]-srtdb[i]]    
    return distances


# Quantifies the quality of the merging
def ordinality(bins):
    single_bin = [sum(bins, [])]
    baseline_dists = get_distances(single_bin)
    new_dists = get_distances(bins)

    ord_sum = 0
    for v in baseline_dists:
        if not new_dists[v][0] and not new_dists[v][1]:
            ord_sum += 1
        elif not new_dists[v][0] and new_dists[v][1]:
            ord_sum += baseline_dists[v][1]/new_dists[v][1]
        elif new_dists[v][0] and not new_dists[v][1]:
            ord_sum += baseline_dists[v][0]/new_dists[v][0]
        else:
            ord_sum += 1/2*baseline_dists[v][0]/new_dists[v][0] + 1/2*baseline_dists[v][1]/new_dists[v][1]

    return ord_sum/len(baseline_dists)


# Does this bin-merging arrangement fit our threshold constraint?
def checkConstraint(bins, value_occurs, threshold):
    bins_with_values = []
    for bin in bins:
        bins_with_values.append([value_occurs[value] for value in bin])

    for bin in bins_with_values:
        tmp = sum(bin)
        if tmp < threshold and len(bin)!=0:
            return False
    return True


def binCounts(bins, value_occurs, i, threshold, map):
   
    if i == len(value_occurs):
        if checkConstraint(bins, value_occurs, threshold):
            prob = max_sampling_prob(bins)
            ord = ordinality(bins)
            no_bins = len([bin for bin in bins if bin != []])

            if prob not in distinct_probs or True:
                distinct_probs[prob] = bins
                occur_bins = [[value_occurs[value] for value in bin] for bin in bins]
                print(f'{no_bins:6d}  |  {ord:10f}  |  {prob:20f}  |  {str(bins):50s}  |  {occur_bins}')
        return
    
    for bin in bins:
        bin.append(map[i])
        binCounts(bins, value_occurs, i+1, threshold, map)
        bin.pop()


if __name__ == "__main__":
   
    k = 1.2
    samples = 1000

    # Generate dataset
    values = sorted([random.choice(range(30,34)) for _ in range(samples)])
    print("\n\nGenerated random data.")
    

    # Get value occurrences
    value_occurs = dict((i, values.count(i)) for i in values)
    print("We have", len(value_occurs), "unique values.")
    average = sum(value_occurs.values())/len(value_occurs.values())
    print("Min occurrences:", min(value_occurs.values()), "   |   Max occurrences:", max(value_occurs.values()), "   |   Average occurrences:", average, "\n\n")

    # Fit these values in corresponding bins
    bins = [[] for i in range(len(value_occurs.values()))]

    # Helper dict (correspondance between the value_occurs dict and the bins generated above
    map = {}
    position = 0
    for v in value_occurs:
        map[position] = v
        position += 1

    #Find the ordinality on the dataset before merging (1 bin per value)
    #single_bin = [list(value_occurs.keys())]
    #single_bin_distances = get_distances(single_bin)    
    #print(single_bin_distances)
    #uniform_bins = [[v] for v in value_occurs]
    #uniform_distances = get_distances(uniform_bins)
    #print(uniform_distances)
  
    print(f'{"#Values"}  |  {"Ordinality":10s}  |  {"Max sampling prob":20s}  |  {"Bins (Values)":50s}  |  {"Bins (Occurrences/Counts)"}')
    binCounts(bins, value_occurs, 0, k*average, map); # Enumerate all possible bin merges














