from itertools import combinations # permutation
from scipy.stats import normaltest, ttest_ind, mannwhitneyu, bartlett, levene

from experiments_eval import get_metric_list


def normal_distribution(x: list) -> bool:
    """
    Test whether a sample differs from a normal distribution.
    This function tests the null hypothesis that a sample comes from a normal distribution. It is based on D’Agostino
    and Pearson’s test that combines skew and kurtosis to produce an omnibus test of normality.
    :param x: sample data
    :return:If False, null hypothesis is rejected and it can be assumed that the sample come from a normal distribution
            If True, null hypothesis cannot be rejected and it can be assumed that the sample are not normal
            distributed
    """
    k2, p = normaltest(x)
    alpha = 0.05
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        # The null hypothesis can be rejected
        return False
    else:
        # The null hypothesis cannot be rejected
        return True


def bartlett_test(a: list, b: list, alpha: float) -> bool:
    """
    Perform Bartlett’s test for equal variances.
    Bartlett’s test tests the null hypothesis that all input samples are from populations with equal variances.
    For samples from significantly non-normal populations, Levene’s test levene is more robust.s.
    :param a: sample data one
    :param b: sample data two
    :param alpha: corrected alpha value (according to Bonferroni)
    :return: If False, null hypothesis is rejected and it can be assumed that the variances are equal across all samples
            If True, null hypothesis cannot be rejected and it can be assumed that the variances are not equal across
            all samples
    """
    stat, p = bartlett(a, b)
    if p < alpha:  # null hypothesis: the variances are equal across all samples/groups
        # The null hypothesis can be rejected
        return False
    else:
        # The null hypothesis cannot be rejected
        return True


def levene_test(a: list, b: list, alpha: float) -> bool:
    """
    Perform Levene test for equal variances.
    The Levene test tests the null hypothesis that all input samples are from populations with equal variances. Levene’s
    test is an alternative to Bartlett’s test bartlett in the case where there are significant deviations from normality.
    :param a: sample data one
    :param b: sample data two
    :param alpha: corrected alpha value (according to Bonferroni)
    :return: If False, null hypothesis is rejected and it can be assumed that the variances are equal across all samples
            If True, null hypothesis cannot be rejected and it can be assumed that the variances are not equal across
            all samples
    """
    stat, p = levene(a, b)
    if p < alpha:  # null hypothesis: the variances are equal across all samples/groups
        # The null hypothesis can be rejected
        return False
    else:
        # The null hypothesis cannot be rejected
        return True


def student_t_test(a: list, b: list, equal_var: bool, alpha: float) -> bool:
    """
    Calculate the T-test for the means of two independent samples of scores.
    This is a test for the null hypothesis that 2 independent samples have identical average (expected) values.
    :param a: sample data one
    :param b: sample data two
    :param equal_var: If True, perform a standard independent 2 sample test that assumes equal population variances.
                    If False, perform Welch’s t-test, which does not assume equal population variance.
    :param alpha: corrected alpha value (according to Bonferroni)
    :return: If False, null hypothesis is rejected and it can be assumed that the difference in the samples is
            statistically significant.
            If True, null hypothesis cannot be rejected and it can be assumed the result of a statistical coincidence.
    """
    t_stat, p = ttest_ind(a, b, axis=0, equal_var=equal_var, alternative='two-sided')
    if p < alpha:  # null hypothesis: a, b
        # The null hypothesis can be rejected"
        return False
    else:
        # The null hypothesis cannot be rejected
        return True


def mann_whitney_u_test(a: list, b: list, alpha: float) -> bool:
    """
    Perform the Mann-Whitney U rank test on two independent samples.
    The Mann-Whitney U test is a nonparametric test of the null hypothesis that the distribution underlying sample x is
    the same as the distribution underlying sample y. It is often used as a test of difference in location between
    distributions.
    :param a: sample data one
    :param b: sample data two
    :param alpha: corrected alpha value (according to Bonferroni)
    :return: If False, null hypothesis is rejected and it can be assumed that the difference in the samples is
            statistically significant.
            If True, null hypothesis cannot be rejected and it can be assumed the result of a statistical coincidence.
    """
    U1, p = mannwhitneyu(a, b, alternative='two-sided', method="exact")

    if p < alpha:  # null hypothesis: a, b
        # The null hypothesis can be rejected"
        return False
    else:
        # The null hypothesis cannot be rejected
        return True


def significance_test(experiment_a: str, experiment_b: str, metric_key: str, alpha: float) -> None:
    """

    :param experiment_a: name of the experiment 'model_season'
    :param experiment_b: name of the experiment 'model_season'
    :param metric_key: metric of the experiments to be investigated.
    :param alpha: corrected alpha value (according to Bonferroni)
    :return:
    """
    metrics_a = get_metric_list(experiment_name=experiment_a, metric_key=metric_key)
    metrics_b = get_metric_list(experiment_name=experiment_b, metric_key=metric_key)

    if "rls" in experiment_a or "rls" in experiment_b:
        if len(metrics_a) == 1:
            metrics_a = metrics_a * 30
        else:
            metrics_b = metrics_b * 30
    assert len(metrics_a) == len(metrics_b), "sample sizes are not equal."

    norm_dist_a = normal_distribution(metrics_a)
    norm_dist_b = normal_distribution(metrics_b)

    if norm_dist_a and norm_dist_b:
        # T- test
        test = "T-test"
        equal_var = bartlett_test(metrics_a, metrics_b, alpha)
        h0 = student_t_test(metrics_a, metrics_b, equal_var, alpha)
    else:
        # Mann Whitney U Test
        test = "Mann Whitney U Test"
        h0 = mann_whitney_u_test(metrics_a, metrics_b, alpha)

    if not h0:
        print(f"{test}: {experiment_a} and {experiment_b} are statistically significant.")
    else:
        print(f"{test}: {experiment_a} and {experiment_b} are based on a statistical coincidence.")


def main():
    ml = "offline" # online
    season = "all" # summer, winter

    offline_experiments = ["lr", "krr", "svr", "rfr", "dnn", "cnn", "lstm"]
    online_experiments = ["rls", "sgd", "odl", "hbp", "odl_er"]
    if ml == "offline":
        experiments = offline_experiments
        metric_key = "rmse"
    else:
        experiments = online_experiments
        metric_key = "rmse_24"

    comb = combinations(experiments, 2)
    comb_list = [list(item) for item in comb]

    # alpha_adj = alpha/ number of experiments
    alpha = 0.05
    alpha_adj = alpha / len(comb_list)

    for c in comb_list:
        experiment_a = f"{c[0]}_{season}"
        experiment_b = f"{c[1]}_{season}"
        significance_test(experiment_a, experiment_b, metric_key, alpha_adj)


if __name__ == "__main__":
    main()
