import numpy as np


class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN."""

    def __init__(self, data, output_info, log_frequency):
        self._data = data
        
        '''
        def is_discrete_column(column_info):
            return (len(column_info) == 1
                    and column_info[0].activation_fn == "softmax")
        '''
        n_discrete_columns = sum([1 for column_info in output_info])
        self._discrete_column_matrix_st = np.zeros(n_discrete_columns, dtype="int32")

        # Store the row id for each category in each discrete column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the
        # a-th discrete column (discrete_column_id) equal value b (index).
        self._rid_by_cat_cols = []
        #import sys
        #np.set_printoptions(threshold=sys.maxsize)
    
        # Compute _rid_by_cat_cols
        discrete_ind = 0
        st = 0
        for column_info in output_info:
            if len(column_info) == 1: #Discrete (softmax)
                self._discrete_column_matrix_st[discrete_ind] = st #Map feature-columns to their one hot encoding
                discrete_ind += 1                

                span_info = column_info[0]                
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0]) #Get the indices of the rows where st+j (column_id) is non-zero, so that we can sample later
                    

                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            elif len(column_info) == 2: #Discretized (tanh + softmax)
                st += 1
                self._discrete_column_matrix_st[discrete_ind] = st #Map feature-columns to their one hot encoding
                discrete_ind += 1                
               
                span_info = column_info[1]            
                ed = st + sum([span_info.dim for span_info in column_info]) - 1   #(1 for tanh + N dim for the means)
               
                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0]) #Get the indices of the rows where st+j (column_id) is non-zero, so that we can sample later

                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                raise Exception("Shouldn't happen #1!")

        assert st == data.shape[1]

        # Prepare an interval matrix for efficiently sample conditional vector
        max_category = 0
        for column_info in output_info:
            if len(column_info)==1:
                max_category = max(max_category, column_info[0].dim) #(softmax)
            else:
                max_category = max(max_category, column_info[1].dim) #(tanh, softmax)
        
        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        
        self._n_categories = 0
        for column_info in output_info:
            if len(column_info)==1:
                self._n_categories += column_info[0].dim #(softmax)
            else:
                self._n_categories += column_info[1].dim #(tanh, softmax)

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if len(column_info)==1:
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                         
                assert category_prob.shape == self._discrete_column_category_prob[current_id, :span_info.dim].shape
                self._discrete_column_category_prob[current_id, :span_info.dim] = (category_prob)
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            elif len(column_info)==2:
                span_info = column_info[1]
                st += 1
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                
                assert category_prob.shape == self._discrete_column_category_prob[current_id, :span_info.dim].shape
                self._discrete_column_category_prob[current_id, :span_info.dim] = (category_prob)
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                raise Exception("Shouldn't happen #2!")
            
    def _random_choice_prob_index(self, discrete_column_id):
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def sample_condvec(self, batch):
        """Generate the conditional vector for training.

        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        if self._n_discrete_columns == 0:
            return None

        discrete_column_id = np.random.choice(np.arange(self._n_discrete_columns), batch)
        category_id_in_col = self._random_choice_prob_index(discrete_column_id)

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        
        mask[np.arange(batch), discrete_column_id] = 1
        
        category_id = (self._discrete_column_cond_st[discrete_column_id] + category_id_in_col)
        cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col

    def sample_original_condvec(self, batch):
        """Generate the conditional vector for generation use original frequency."""
        if self._n_discrete_columns == 0:
            return None

        cond = np.zeros((batch, self._n_categories), dtype='float32')

        for i in range(batch):
            row_idx = np.random.randint(0, len(self._data))
            col_idx = np.random.randint(0, self._n_discrete_columns)
            matrix_st = self._discrete_column_matrix_st[col_idx]
            matrix_ed = matrix_st + self._discrete_column_n_category[col_idx]
            pick = np.argmax(self._data[row_idx, matrix_st:matrix_ed])
            cond[i, pick + self._discrete_column_cond_st[col_idx]] = 1

        return cond

    def sample_data(self, n, col, opt):
        """Sample data from original training data satisfying the sampled conditional vector.

        Returns:
            n rows of matrix data.
        """
        if col is None:
            idx = np.random.randint(len(self._data), size=n)
            return self._data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))
        return self._data[idx]
    
    def sample_data_quantum(self, n, col, opt, value_maps_id):
        """Sample data from original training data satisfying the sampled conditional vector.

        Returns:
            n rows of matrix data.
        """
        #if col is None:
        #    idx = np.random.randint(len(self._data), size=n)
        #    return self._data[idx]

        samples = []
        idx = []
        
        # Get alternative opts
        for c, o in zip(col, opt):            
            #print("-----------\n", "c", c, "o", o)
            
            values,freqs = None, None
            if c in value_maps_id:
                #Get the other values that are entagled with c,o
                #print("discrete")
                #print(value_maps_id[c])
                values, freqs = zip(*value_maps_id[c][o])
            else:
                assert false, "Column " + str(c) + " not in value_maps_id"
                
            #print("values", values)
            sampling_pool = []
            for v_idx in values: #The sampling pool contains all the samples with value either o or a value entangled with o
                sampling_pool.extend(self._rid_by_cat_cols[c][v_idx]) # _rid_by_cat_cols[c][v] is a list of all rows with the c-th discrete column equal value v (index).
                
            tmp_sample_idx = np.random.choice(sampling_pool) # Pick one sample from the pool of ids
            tmp_sample = self._data[tmp_sample_idx] #Get the sample corresponding to that id
            #print("tmp_sample", tmp_sample)
            
            #Make sure that one of the columns in the list is 1.0. This is to assert that the pool sampling works as intended.
            st = self._discrete_column_matrix_st[c]
            assert sum([tmp_sample[st+v_ind] for v_ind in values]) >= 1.0, "tmp_sample with those was "+str([tmp_sample[st+v_ind] for v_ind in values])

            #We now flip the value in column a based on the entanglement probability
            #Switch all the opts under the col to 0.0
            #and then switch one of them to 1.0
            for v in values:
                tmp_sample[st+v] = 0.0

            #values, freqs = zip(*value_maps_id[c][o])
            val = np.random.choice(values, p=freqs)
            tmp_sample[st+val] = 1.0

            #print("sample value", tmp_sample[st+val])
            samples.append(tmp_sample)

        return np.array(samples)

    def dim_cond_vec(self):
        return self._n_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id = self._discrete_column_matrix_st[condition_info["discrete_column_id"]
                                             ] + condition_info["value_id"]
        vec[:, id] = 1
        return vec