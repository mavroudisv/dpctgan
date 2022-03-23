import random
import pandas as pd

DEMO_URL = 'http://ctgan-data.s3.amazonaws.com/census.csv.gz'
THRESHOLD      = 1000

distinct_probs = {}
SOLUTIONS      = []


def load_demo():
    return pd.read_csv(DEMO_URL, compression='gzip')

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

def get_min_distance(lst):
    lst = lst[0]
    lst.sort()
    min_distance = float('inf')
    for i in range(1,len(lst)):
        min_distance = min(min_distance, lst[i] - lst[i-1])
    #assert(min_distance>=1)
    return min_distance

# Quantifies the quality of the merging
def ordinality_min(bins):
    single_bin = [sum(bins, [])]
    baseline_dist = get_min_distance(single_bin)
    new_dists = get_distances(bins)

    ord_sum = 0
    for v in new_dists:
        if not new_dists[v][0] and not new_dists[v][1]:
            ord_sum += 1
        elif not new_dists[v][0] and new_dists[v][1]:
            ord_sum += baseline_dist/new_dists[v][1]
        elif new_dists[v][0] and not new_dists[v][1]:
            ord_sum += baseline_dist/new_dists[v][0]
        else:
            ord_sum += 1/2*baseline_dist/new_dists[v][0] + 1/2*baseline_dist/new_dists[v][1]
    return ord_sum/len(new_dists)


# Quantifies the quality of the merging
def ordinality_mean(bins):
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
            ord = ordinality_min(bins)
            no_bins = len([bin for bin in bins if bin != []])

            if prob not in distinct_probs or True:
                distinct_probs[prob] = bins
                occur_bins = [[value_occurs[value] for value in bin] for bin in bins]
                #print(f'{no_bins:6d}  |  {ord:10f}  |  {prob:20f}  |  {str(bins):50s}  |  {occur_bins}')
                SOLUTIONS.append([no_bins, ord, prob, str(bins), occur_bins])

        return
    
    for bin in bins:
        bin.append(map[i])
        binCounts(bins, value_occurs, i+1, threshold, map)
        bin.pop()


if __name__ == "__main__":

    # Generate dataset
    # samples = 1000
    #values = sorted([random.choice(range(30,34)) for _ in range(samples)])
    #print("\n\nGenerated random data.")

    #Load dataset
    values = load_demo()['age'].tolist()
   

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
  
    
    binCounts(bins, value_occurs, 0, 1000, map); # Enumerate all possible bin merges
    
    solutions_sort_by_sampling = sorted(SOLUTIONS, key=lambda tup: tup[2])
    solutions_filtered = filter(lambda solution: solution[2] < THRESHOLD, solutions_sort_by_sampling)
    solutions_filtered_sorted = sorted(solutions_filtered, key=lambda tup: tup[1], reverse=True)
    
    print(f'{"#Values"}  |  {"Ordinality":10s}  |  {"Max sampling prob":20s}  |  {"Bins (Values)":50s}  |  {"Bins (Occurrences/Counts)"}')    
    for s in solutions_filtered_sorted:
        if s[2] <= THRESHOLD:
            print(f'{s[0]:6d}  |  {s[1]:10f}  |  {s[2]:20f}  |  {str(s[3]):50s}  |  {s[4]}')













