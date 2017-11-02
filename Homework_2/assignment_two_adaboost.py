# Assignment 2, Part 3: Adaboost
#
# Version 0.1

import math
import numpy as np
from assignment_two_svm \
    import evaluate_classifier, print_evaluation_summary


# TASK 3.1
# Complete the function definition below
# Remember to return a function, not the
# sign, feature, threshold triple
def weak_learner(instances, labels, dist):

    """ Returns the best 1-d threshold classifier.

    A 1-d threshold classifier is of the form

    lambda x: s*x[j] < threshold

    where s is +1/-1,
          j is a dimension
      and threshold is real number in [-1, 1].

    The best classifier is chosen to minimize the weighted misclassification
    error using weights from the distribution dist.

    """

    dim = len(instances[0])
    num_steps = 500
    s = np.ones(dim)
    error = np.zeros(dim)
    theta = np.zeros(dim)

    for j in range(dim):
        thresh_values = sorted(set(instances[:, j]))
        range_max = max(thresh_values)
        range_min = min(thresh_values)
        step_size = (range_max - range_min) / num_steps
        thresh_values = [range_min + k * step_size for k in range(num_steps)]
        num_thresh_values = len(thresh_values)
        weightedErr = np.zeros((2, num_thresh_values))
        weightedErr[0, :] = [((-instances[:, j] < thresh_values[k]) != labels).dot(dist) for k in range(num_thresh_values)]
        weightedErr[1, :] = [((instances[:, j] < thresh_values[k]) != labels).dot(dist) for k in range(num_thresh_values)]
        thresh_min_index = weightedErr.argmin()
        if thresh_min_index < num_thresh_values: s[j] = -1
        error[j] = weightedErr.flatten()[thresh_min_index]
        theta[j] = thresh_values[np.unravel_index(weightedErr.argmin(), weightedErr.shape)[1]]

    best_index = error.argmin()

    return lambda x: (s[best_index] * x[best_index]) < theta[best_index]


# TASK 3.2
# Complete the function definition below
def compute_error(h, instances, labels, dist):

    """ Returns the weighted misclassification error of h.

    Compute weights from the supplied distribution dist.
    """

    case_labels = np.apply_along_axis(h, 1, instances)
    error_case = (case_labels != labels) * 1

    return dist.dot(error_case)


# TASK 3.3
# Implement the Adaboost distribution update
# Make sure this function returns a probability distribution
def update_dist(h, instances, labels, dist, alpha):

    """ Implements the Adaboost distribution update. """

    case_labels = np.apply_along_axis(h, 1, instances)
    n = len(instances)
    new_dist = [dist[i] * np.exp(-alpha) if case_labels[i] == labels[i] else dist[i] * np.exp(alpha) for i in range(n)]
    new_dist = new_dist / sum(new_dist)

    return new_dist


def run_adaboost(instances, labels, weak_learner, num_iters=20):

    n, d = instances.shape
    n1 = labels.size

    if n1 != n:
        raise Exception('Expected same number of labels as no. of rows in \
                        instances')

    alpha_h = []

    dist = np.ones(n)/n

    for i in range(num_iters):

        print "Iteration: %d" % i
        h = weak_learner(instances, labels, dist)

        error = compute_error(h, instances, labels, dist)

        if error > 0.5:
            break

        alpha = 0.5 * math.log((1-error)/error)

        dist = update_dist(h, instances, labels, dist, alpha)

        alpha_h.append((alpha, h))

    # TASK 3.4
    # return a classifier whose output
    # is an alpha weighted linear combination of the weak
    # classifiers in the list alpha_h
    def classifier(point):
        """ Classifies point according to a classifier combination.

        The combination is stored in alpha_h.
        """

        alpha = np.array(alpha_h)[:, 0]
        h = np.array(alpha_h)[:, 1]
        n = len(alpha)
        weak_classifier_result = np.asarray([h[i](point) for i in range(n)])

        return alpha.dot(2 * weak_classifier_result - 1) > 0

    return classifier


def main():
    data_file = 'ionosphere.data'

    data = np.genfromtxt(data_file, delimiter=',', dtype='|S10')
    instances = np.array(data[:, :-1], dtype='float')
    labels = np.array(data[:, -1] == 'g', dtype='int')

    n, d = instances.shape
    nlabels = labels.size

    if n != nlabels:
        raise Exception('Expected same no. of feature vector as no. of labels')

    train_data = instances[:200]  # first 200 examples
    train_labels = labels[:200]  # first 200 labels

    test_data = instances[200:]  # example 201 onwards
    test_labels = labels[200:]  # label 201 onwards

    print 'Running Adaboost...'
    adaboost_classifier = run_adaboost(train_data, train_labels, weak_learner)
    print 'Done with Adaboost!\n'

    confusion_mat = evaluate_classifier(adaboost_classifier, test_data,
                                        test_labels)
    print_evaluation_summary(confusion_mat)

if __name__ == '__main__':
    main()
