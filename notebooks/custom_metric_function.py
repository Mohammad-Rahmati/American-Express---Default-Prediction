class AmexMetric:
    list_of_act = []
    list_of_pred = []

    def map(self, predicted, actual, weight, offset, model):

        y = actual[0]
        p = predicted[2]
        self.list_of_act.append(y)
        self.list_of_pred.append(p)

        return [1, 1]

    def reduce(self, left, right):
        return [left[0] + right[0], left[1] + right[1]]

    def metric(self, last):

        act = self.list_of_act
        pred = self.list_of_pred
        mirror_act = [1.0 if x == 0.0 else -1.0 for x in act]
        sorted_data = [0.0 if n == 1.0 else 1.0 for _,n in sorted(zip(pred, mirror_act), reverse=True)]
        weight = [20.0 - i * 19.0 for i in sorted_data]
        sum_weight = sum(weight)
        four_pct_cutoff = int(sum_weight * 0.04)

        weight_cumsum_v = 0.0
        cum_pos_found = 0.0
        weight_cumsum = []
        random = []
        cum_pos_found_list = []
        lorentz = []
        gini = []

        total_pos = sum(sorted_data)
        for indx, weight_v in enumerate(weight):
            
            weight_cumsum_v += weight_v
            weight_cumsum.append(weight_cumsum_v)

            random_v = weight_cumsum_v/sum_weight
            random.append(random_v)

            cum_pos_found += sorted_data[indx] * weight_v
            cum_pos_found_list.append(cum_pos_found)

            lorentz_v = cum_pos_found/total_pos
            lorentz.append(lorentz_v)

            gini_v = (lorentz_v - random_v) * weight_v
            gini.append(gini_v)

        total_neg = len(sorted_data) - total_pos
        gini_max = 10.0 * total_neg * (total_pos + 20.0 * total_neg - 19.0) / (total_pos + 20.0 * total_neg)

        indx_cutoff = sum(map(lambda x : x <= four_pct_cutoff, weight_cumsum))

        d = 1.0 * sum(sorted_data[:indx_cutoff]) / total_pos
        g_not_normalized = 1.0 * sum(gini)

        score = 0.5 * (d + g_not_normalized / gini_max)
        
        with open('log.txt', 'a') as f:

            f.write(str(score))
            f.write('\n')
            f.close()

        return score