from collections import namedtuple

import numpy as np
import pandas as pd
from rdt.transformers import OneHotEncodingTransformer
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

##### Privacy ######
from numpy import random

def _set_parameters(self, params):
    ##### Sorting the parameters based on the means ######
    # Sorting makes mergine means much easier and we don't have to deal with indices
    (tmp_weights_, tmp_means_, tmp_covariances_, tmp_precisions_cholesky_) = params
    
    s = pd.Series(np.array(tmp_means_)[:,0]).sort_values()
    original_indices = s.index.tolist()
    
    self.weights_ = np.asarray([tmp_weights_[i] for i in original_indices])
    self.means_ = np.asarray([tmp_means_[i] for i in original_indices])
    self.covariances_ = np.asarray([tmp_covariances_[i] for i in original_indices])
    self.precisions_cholesky_ = np.asarray([tmp_precisions_cholesky_[i] for i in original_indices])
    
    #print("means_ ", self.means_)
    #print("weights_ ", self.weights_)
    #print("covariances_ ", self.covariances_)
    #print("precisions_cholesky_ ", self.precisions_cholesky_)
    
    ##### Sorting the parameters based on the means ######

    # Attributes computation
    _, n_features = self.means_.shape

    if self.covariance_type == 'full':
        self.precisions_ = np.empty(self.precisions_cholesky_.shape)
        for k, prec_chol in enumerate(self.precisions_cholesky_):
            self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

    elif self.covariance_type == 'tied':
        self.precisions_ = np.dot(self.precisions_cholesky_,
                                  self.precisions_cholesky_.T)
    else:
        self.precisions_ = self.precisions_cholesky_ ** 2

GaussianMixture._set_parameters = (_set_parameters) #Set parameters sorted

##### Privacy ######

SpanInfo = namedtuple("SpanInfo", ["dim", "activation_fn"])
ColumnTransformInfo = namedtuple(
    "ColumnTransformInfo", ["column_name", "column_type",
                            "transform", "transform_aux",
                            "output_info", "output_dimensions"])

def get_occurences_counts(data, add_noise=False, noise_d_mean=None, noise_d_std=None):
    counts = {}
    for feature in data.head():
        #print(list(counts[feature].items()))
        #for key, row in counts[feature].iteritems():
            #counts[feature][key] = row + random.normal(loc=10, scale=std) #Random values here for now, just making sure we won't go into the negatives
        counts[feature] = get_occurences_counts_per_feature(data[feature],add_noise,noise_d_mean,noise_d_std)
    return counts

def get_occurences_counts_per_feature(data_feature, add_noise=False, noise_d_mean=None, noise_d_std=None):
    counts_feature = {}
    counts_feature = data_feature.value_counts()

    # Make them noisy counts
    if add_noise:
        for key, row in counts_feature.iteritems():
            counts_feature[key] = row + random.normal(loc=noise_d_mean, scale=noise_d_std) #Random values here for now, just making sure we won't go into the negatives
    return counts_feature

 
def get_entanglements_per_feature(occurs, l_threshold):
    values_above = []
    values_below = []
    
    #Get values below L
    for label, count in occurs.iteritems():
        #print(label, count)
        if count > l_threshold:
            values_above.append((label, count))
        else:
            values_below.append((label, count))

    
    ## Sort based on occurences
    values_below = sorted(values_below, key=lambda tup: tup[1], reverse=True)
    values_above = sorted(values_above, key=lambda tup: tup[1], reverse=True)
    
    value_entanglement_map = {} # Map of the superpositions
    for index, _ in enumerate(values_above):
        above_tuple = values_above[index] #(value, count)
        position = [(above_tuple[0], 1)]  #value, probability
        value_entanglement_map[above_tuple[0]] = position
    
    assert len(values_above) > 0
    assert len(values_above) >= len(values_below) #Needed because of the merging algorithm used below

    # MERGING ALGORITHM #
    # Entangle some of the "above" values with some values with occurences "below" the threshold    
    ## Todo: This insanely simple "algorithm" assumes that we have more values above the threshold than below.
    ## Come up with a nice, generic algorithm for deciding what to merge
    for index, _ in enumerate(values_below):
        above_tuple = values_above[index]
        below_tuple = values_below[index]
        
        count_sum = above_tuple[1] + below_tuple[1]
        ratio_above = above_tuple[1] / count_sum
        ratio_below = below_tuple[1] / count_sum
        
        superposition = [(above_tuple[0], ratio_above), (below_tuple[0], ratio_below)]
        value_entanglement_map[above_tuple[0]] = superposition
        value_entanglement_map[below_tuple[0]] = superposition
    
    return value_entanglement_map

def get_entanglements(occurs, l_threshold):
    value_maps = {}
    for feature in occurs.keys():
        value_maps[feature] = get_entanglements_per_feature(occurs[feature], l_threshold)
    return value_maps
    
def convert_names_to_ids(transf, value_maps):
    value_maps_id = {}
    for name in value_maps.keys():
        info = transf.convert_column_name_to_id(name)
        id = info['discrete_column_id'] #It's a bit surprising but the sampler prefers discrete_column_ids instead of generic column_ids
        value_maps_id[id] = value_maps[name]
    return value_maps_id
#### Privacy #### 

class DataTransformerDP(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def __init__(self, privacy_quantum,
                noisy_occurences_d, 
                noise_d_mean, 
                noise_d_std, 
                l_threshold_d, 
                noisy_occurences_c, 
                noise_c_mean, 
                noise_c_std, 
                l_threshold_c, 
                components_c,
                synthetic_std,
                max_clusters=10, 
                weight_threshold=0.005
                ):
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

        #### Privacy ####
        self.privacy_quantum = privacy_quantum
        
        #Discrete
        self.noisy_occurences_d = noisy_occurences_d
        self.noise_d_mean = noise_d_mean
        self.noise_d_std = noise_d_std
        self.occurences = {}    #Discrete
        self.value_maps = {}    #Discrete
        self.value_maps_id = {} #Discrete
        self.l_threshold_d = l_threshold_d
        
        #Continuous
        self.noisy_occurences_c = noisy_occurences_c
        self.noise_c_mean = noise_c_mean
        self.noise_c_std = noise_c_std
        self.l_threshold_c = l_threshold_c
        self.components_c = components_c
        self.synthetic_std = synthetic_std
        self.components_c_merged = -1 # Is determined later
        #### Privacy ####
        

    def _fit_continuous(self, column_name, column_data):             
        """Train Bayesian GMM for continuous column."""
        raw_column_data = column_data.values
        gm = BayesianGaussianMixture(
            self._max_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1
        )
        gm.fit(raw_column_data.reshape(-1, 1))
        valid_component_indicator = gm.weights_ > self._weight_threshold
        num_components = valid_component_indicator.sum()

        return ColumnTransformInfo(
            column_name=column_name, column_type="continuous", transform=gm,
            transform_aux=valid_component_indicator,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components)
        

    def _fit_continuous_privacy(self, column_name, column_data):
        """Train GMM for continuous column."""      
        raw_column_data = column_data.values
                
      
        assert raw_column_data.shape[0] > self.l_threshold_c #The total number of samples is less than the privacy threshold! Consider needing less privacy or gather more data.
        gm = GaussianMixture(self.components_c)
        gm.fit(raw_column_data.reshape(-1, 1))
        
           
        valid_component_indicator = gm.weights_ > 0
        num_components = valid_component_indicator.sum()

        return ColumnTransformInfo(
            column_name=column_name, 
            column_type="continuous", 
            transform=gm,
            transform_aux=valid_component_indicator,
            output_info=[SpanInfo(1, 'tanh'), 
            SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components
            )

    def _fit_discrete(self, column_name, column_data):
        """Fit one hot encoder for discrete column."""
        ohe = OneHotEncodingTransformer()
        ohe.fit(column_data.values)
        num_categories = len(ohe.dummies)

        #### Privacy ####
        if self.privacy_quantum:
            self.occurences[column_name] = get_occurences_counts_per_feature(column_data, self.noisy_occurences_d, self.noise_d_mean, self.noise_d_std) # Count occurences
            self.value_maps[column_name] = get_entanglements_per_feature(self.occurences[column_name], self.l_threshold_d) # Decide on entanglements       
        #### Privacy ####

        return ColumnTransformInfo(
            column_name=column_name, 
            column_type="discrete", 
            transform=ohe,
            transform_aux=None,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories
            )


    def fit(self, raw_data, discrete_columns=tuple()):
        """Fit GMM for continuous columns and One hot encoder for discrete columns.
        This step also counts the #columns in matrix data, and span information.
        """
  
        self.output_info_list = []
        self.output_dimensions = 0

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            raw_data = pd.DataFrame(raw_data)
        else:
            self.dataframe = True

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info_list = []
  
        
        for column_name in raw_data.columns:
            #raw_column_data = raw_data[column_name].values #CTGAN
            column_data = raw_data[column_name]  #DPCTGAN
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(column_name, column_data)
            else:
                if self.privacy_quantum:
                    column_transform_info = self._fit_continuous_privacy(column_name, column_data) #### Privacy
                else:    
                    column_transform_info = self._fit_continuous(column_name, column_data)

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)
          
        #### Privacy ####
        if self.privacy_quantum:		
            self.value_maps_id = convert_names_to_ids(self, self.value_maps) # Move from "labels" to "column ids"
        #### Privacy ####

    def _transform_continuous(self, column_transform_info, raw_column_data):
        gm = column_transform_info.transform

        valid_component_indicator = column_transform_info.transform_aux
        num_components = valid_component_indicator.sum()

        means = gm.means_.reshape((1, self._max_clusters))
        stds = np.sqrt(gm.covariances_).reshape((1, self._max_clusters))
        normalized_values = ((raw_column_data - means) / (4 * stds)
                             )[:, valid_component_indicator]
        component_probs = gm.predict_proba(
            raw_column_data)[:, valid_component_indicator]

        selected_component = np.zeros(len(raw_column_data), dtype='int')
        for i in range(len(raw_column_data)):
            component_porb_t = component_probs[i] + 1e-6
            component_porb_t = component_porb_t / component_porb_t.sum()
            selected_component[i] = np.random.choice(
                np.arange(num_components), p=component_porb_t)

        selected_normalized_value = normalized_values[np.arange(len(raw_column_data)), selected_component].reshape([-1, 1])
        selected_normalized_value = np.clip(selected_normalized_value, -.99, .99)

        selected_component_onehot = np.zeros_like(component_probs)
        selected_component_onehot[np.arange(len(raw_column_data)),
                                  selected_component] = 1
        return [selected_normalized_value, selected_component_onehot]


    def _merge_values(self, values, counts):
        total_counts  = sum(counts)
        total_value = 0
        for i,v in enumerate(values):
            total_value += v * counts[i]/total_counts
        return (total_value, total_counts)

    #### Privacy ####
    def _transform_continuous_privacy(self, column_transform_info, raw_column_data):
        means_safe = []
        stds_safe = []
        
        gm = column_transform_info.transform

        means_unsafe = gm.means_.reshape((1, self.components_c))[0] #These are not sorted (causes problems as we want to merge neighbouring means)        
        component_mapping = {}#i:[] for i in range(len(means_unsafe))}
        
        #stds_unsafe = np.sqrt(gm.covariances_).reshape((1, self.components_c)) #Not used, useful though
        #print("stds_unsafe", stds_unsafe)
        stds_tmp = [self.synthetic_std] * self.components_c #These are those we use

        valid_component_indicator = column_transform_info.transform_aux
        num_components = valid_component_indicator.sum()       
        component_probs = gm.predict_proba(raw_column_data)[:, valid_component_indicator]
        
        
        ###### Sample Counts ###### 
        #Assign samples to means and count them
        counts_per_component = {c:0 for c in range(self.components_c)}
        selected_component = np.zeros(len(raw_column_data), dtype='int')
        for i in range(len(raw_column_data)):
            component_prob_t = component_probs[i] + 1e-6
            component_prob_t = component_prob_t / component_prob_t.sum()
            
            selected_component[i] = np.random.choice(np.arange(self.components_c), p=component_prob_t)
            counts_per_component[selected_component[i]] += 1

        #Add noise to the sample counts
        for c in range(self.components_c):
            counts_per_component[c] = counts_per_component[c] + random.normal(loc=self.noise_c_mean, scale=self.noise_c_std) #Random values here for now, just making sure we won't go into the negatives
        ###### Sample Counts ###### 

        ###### Means Merging ######         
        last_merge = False
        tmp_means = []  #Only above the threshold means here, but may change over time
        tmp_counts = [] #Only above the threshold counts here, but may change over time
        tmp_below_neighbours_idx = []
        for i in range(len(means_unsafe)):
            # We keep a list of underrepresented means and merge them as soon as one with sufficient count comes along
            
            if counts_per_component[i] >= self.l_threshold_c: #Mean `i` is safe, so we can add it; let's see whether there is something to merge next
                tmp_means.append(means_unsafe[i])
                tmp_counts.append(counts_per_component[i])
                component_mapping[i] = len(tmp_means)-1 #Keep track of the mapping between the old means and the new ones
           
                if tmp_below_neighbours_idx != []: #There are means that couldn't be combined to make it above the threshold
                    #Merge the elements of `tmp_below_neighbours_idx` with whichever safe neighbour their are closer to
                    while tmp_below_neighbours_idx != []:
                        idx = tmp_below_neighbours_idx.pop()
                        tmp_mean = means_unsafe[idx]
                        tmp_count = counts_per_component[idx]

                        #Determine the ID of the closest mean value to merge with
                        merge_with_id = 0 
                        if len(tmp_means) > 1: #There is more than one option, so let's select the best one
                            dist_down = abs(tmp_means[-2] - tmp_mean) # mean for merging down
                            dist_up = abs(tmp_means[-1] - tmp_mean) # mean for merging up
                            
                            if dist_down > dist_up: # if our current mean is closer than the previous (safe) mean we merge up!
                                merge_with_id = len(tmp_means)-1
                            else: # if the previous (safe) mean was closer we merge down
                                merge_with_id = len(tmp_means)-2
                                
                        # We compute the new mean as the weighted average between its previous value and the mean we merged with it
                        (tmp_mean, tmp_count) = self._merge_values([tmp_mean, tmp_means[merge_with_id]], [tmp_count, tmp_counts[merge_with_id]])    
                        tmp_means[merge_with_id] = tmp_mean
                        tmp_counts[merge_with_id] = tmp_count

                        component_mapping[idx] = merge_with_id #Keep track of the mapping between the old means and the new ones
                
            else :  # counts_per_component[i] < self.l_threshold_c:
                tmp_below_neighbours_idx.append(i) #Keep pointer to unsafe means

                #Check the merge candidates list
                neighbour_means = [means_unsafe[n] for n in tmp_below_neighbours_idx]
                neighbour_counts =[counts_per_component[n] for n in tmp_below_neighbours_idx] 
                (neighbour_mean, neighbour_count) = self._merge_values(neighbour_means, neighbour_counts)
                
                if neighbour_counts >= self.l_threshold_c: #We found a combination of "below" means that works
                    tmp_means.append(neighbour_mean)
                    tmp_counts.append(neighbour_count)
                    
                    for idx in tmp_below_neighbours_idx: #Keep track of the mapping between the old means and the new ones
                        component_mapping[idx] = len(tmp_means)-1 
                    tmp_below_neighbours_idx = [] #Flush list
                    
                elif neighbour_counts < self.l_threshold_c and i == len(means_unsafe)-1: #If we are in the last element of the `means_unsafe`, merge with the last mean that was above the threshold
                    assert last_merge == False #Should get here only once!
                    last_merge = True
                    
                    (tmp_mean, tmp_count) = self._merge_values([neighbour_mean, tmp_means[-1]], [neighbour_count, tmp_counts[-1]])
                    tmp_means[-1] = tmp_mean
                    tmp_counts[-1] = tmp_count
                    
                    for idx in tmp_below_neighbours_idx: #Keep track of the mapping between the old means and the new ones
                        component_mapping[idx] = len(tmp_means)-1 
                    
                    tmp_below_neighbours_idx = [] #Flush list
                
                else:  #tmp_below_neighbours_idx doesn't have enough total_counts yet, keep going
                    pass
        
        
        self.means_safe = tmp_means
        self.components_c_merged = len(self.means_safe)
        self.stds_safe = [self.synthetic_std] * self.components_c_merged
        ###### Means Merging ###### 
        
        ###### Sample Remapping ######
        for r in range(len(raw_column_data)):
            selected_component[r] = component_mapping[selected_component[r]]
        ###### Sample Remapping ######            

        means = np.array(self.means_safe).reshape((1, self.components_c_merged))
        stds = np.array(self.stds_safe).reshape((1, self.components_c_merged))
        valid_component_indicator = [True] * self.components_c_merged
        
        #Business as usual from now on
        normalized_values = ((raw_column_data - means) / (4 * stds))[:, valid_component_indicator] #normalized distances from each mean
        selected_normalized_value = normalized_values[np.arange(len(raw_column_data)), selected_component].reshape([-1, 1])
        selected_normalized_value = np.clip(selected_normalized_value, -.99, .99)

        selected_component_onehot = np.zeros_like(component_probs)
        selected_component_onehot[np.arange(len(raw_column_data)),selected_component] = 1
        return [selected_normalized_value, selected_component_onehot]
        
        
    #### Privacy ####

    def _transform_discrete(self, column_transform_info, raw_column_data):
        ohe = column_transform_info.transform
        return [ohe.transform(raw_column_data)]

    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)

        column_data_list = []
        for column_transform_info in self._column_transform_info_list:
            column_data = raw_data[[column_transform_info.column_name]].values
            if column_transform_info.column_type == "continuous":
                if self.privacy_quantum: 
                    column_data_list += self._transform_continuous_privacy(column_transform_info, column_data)  #### Privacy
                else:
                    column_data_list += self._transform_continuous(column_transform_info, column_data)
            else:
                assert column_transform_info.column_type == "discrete"
                column_data_list += self._transform_discrete(
                    column_transform_info, column_data)

        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous_privacy(self, column_transform_info, column_data, sigmas, st):
        #gm = column_transform_info.transform
        valid_component_indicator = column_transform_info.transform_aux

        selected_normalized_value = column_data[:, 0]
        selected_component_probs = column_data[:, 1:]

        if sigmas is not None:
            sig = sigmas[st]
            selected_normalized_value = np.random.normal(selected_normalized_value, sig)

        selected_normalized_value = np.clip(selected_normalized_value, -1, 1)
        
        if privacy_quantum:
            component_probs = np.ones((len(column_data), self.components_c)) * -100
        else:
            component_probs = np.ones((len(column_data), self._max_clusters)) * -100
        component_probs[:, valid_component_indicator] = selected_component_probs

        means = self.means #gm.means_.reshape([-1])
        stds = self.stds #np.sqrt(gm.covariances_).reshape([-1])
        selected_component = np.argmax(component_probs, axis=1)

        std_t = stds[selected_component]
        mean_t = means[selected_component]
        column = selected_normalized_value * 4 * std_t + mean_t

        return column

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        valid_component_indicator = column_transform_info.transform_aux

        selected_normalized_value = column_data[:, 0]
        selected_component_probs = column_data[:, 1:]

        if sigmas is not None:
            sig = sigmas[st]
            selected_normalized_value = np.random.normal(selected_normalized_value, sig)

        selected_normalized_value = np.clip(selected_normalized_value, -1, 1)
        
        if privacy_quantum:
            component_probs = np.ones((len(column_data), self.components_c)) * -100
        else:
            component_probs = np.ones((len(column_data), self._max_clusters)) * -100
        component_probs[:, valid_component_indicator] = selected_component_probs

        means = gm.means_.reshape([-1])
        stds = np.sqrt(gm.covariances_).reshape([-1])
        selected_component = np.argmax(component_probs, axis=1)

        std_t = stds[selected_component]
        mean_t = means[selected_component]
        column = selected_normalized_value * 4 * std_t + mean_t

        return column

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        return ohe.reverse_transform(column_data)

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]

            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st)
            else:
                assert column_transform_info.column_type == 'discrete'
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data)

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes))
        if not self.dataframe:
            recovered_data = recovered_data.values

        return recovered_data

    def convert_column_name_to_id(self, column_name):
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == "discrete":
                discrete_counter += 1
            column_id += 1    
        return {
            "discrete_column_id": discrete_counter,
            "column_id": column_id,
        }

    def convert_column_name_value_to_id(self, column_name, value):
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == "discrete":
                discrete_counter += 1
            column_id += 1    
        return {
            "discrete_column_id": discrete_counter,
            "column_id": column_id,
            "value_id": np.argmax(column_transform_info.transform.transform(np.array([value]))[0])
        }
