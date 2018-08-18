import numpy as np
import pandas as pd
from hypothesis import given, settings as hyp_settings, strategies

import scikit_posthocs as sp

max_num_groups = 10
min_num_groups = 2
max_group_size = 2000
min_group_size = 10
upper_lim_value = 1000

alpha_signif_level = 0.05

# num_groups = np.random.randint(min_num_groups, max_num_groups)
# group_size = np.random.randint(min_group_size, max_group_size)
# total_num_samples = num_groups*group_size # group_sizes.sum()

method_list = ['posthoc_nemenyi', ]


def make_identical_groups(group_size, num_groups):
    """"""

    # TODO input must be 'mean rank sums' for some methods
    rand_series = np.random.randint(1, group_size, (group_size, 1))

    same_samples = np.tile(rand_series, (num_groups, 1)).squeeze()
    group_membership = np.vstack([np.tile(ix + 1, (group_size, 1)) for ix in np.arange(
        num_groups)])  # np.random.choice(np.arange(num_groups)+1, (group_size, 1))
    group_membership = group_membership.squeeze()

    df = pd.DataFrame(dict(values=same_samples, group=group_membership))

    return df


def upper_diag(array):
    if array.shape[0] == array.shape[1]:
        return array[np.triu_indices_from(array, 1)]
    else:
        raise ValueError('input matrix is not square! size: {}'.format(array.shape))


@hyp_settings(max_examples=1000,
              min_satisfying_examples=100)
@given(strategies.sampled_from(method_list),
       strategies.integers(min_group_size, max_group_size),
       strategies.integers(min_num_groups, max_num_groups))
def test_identical_data_tests_nonsignificant(method_name, group_size, num_groups):
    """Sanity check 0 : same groups should lead to no significance"""

    df = make_identical_groups(group_size, num_groups)
    posthoc_func = getattr(sp, method_name)
    results = posthoc_func(df, val_col='values', group_col='group')

    if np.any(upper_diag(results.as_matrix()) < alpha_signif_level):
        raise ValueError('significant result obtained for identical groups!!')


print()
