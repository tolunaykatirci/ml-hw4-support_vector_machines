# CSE4088 Homework #4
# Tolunay Katirci - 150115014

import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold

# read training and testing data
training_data = np.loadtxt('data/features.train')
test_data = np.loadtxt('data/features.test')

# training set
training_x = training_data[:, 1:]
training_y = training_data[:, 0]
# test set
testing_x = test_data[:, 1:]
testing_y = test_data[:, 0]


def update_y(y, cls):
    # update y with 1s and -1s
    y_updated = np.ones(len(y))
    for i in range(len(y)):
        # make -1 if pointed with index value is not equal to classifier value
        if y[i] != cls:
            y_updated[i] = -1
    return y_updated


def apply_svm(x, y, cls=0, C=0.01, Q=2, log=True):
    # if cls!=-1, update y by selected classifier value
    if cls != -1:
        y = update_y(y, cls)

    # apply SVC method with appropriate values and fit with x and y
    classifier = svm.SVC(kernel='poly', C=C, degree=Q, shrinking=False, gamma=1, coef0=1)
    classifier.fit(x, y)
    # make a prediction for y
    y_predict = classifier.predict(x)
    # calculate E_in
    E_in = np.sum(0 > y * y_predict) / (1.0 * y.size)

    if log: print('C=%f, Q=%d, classifier value=%d, E_in=%f ' % (C, Q, cls, E_in))
    return E_in, classifier, classifier.support_vectors_.shape[0]


def question_2():
    # run smv with wanted classifier values
    apply_svm(training_x, training_y, cls=0)
    apply_svm(training_x, training_y, cls=2)
    apply_svm(training_x, training_y, cls=4)
    apply_svm(training_x, training_y, cls=6)
    apply_svm(training_x, training_y, cls=8)


def question_3():
    # run smv with wanted classifier values
    apply_svm(training_x, training_y, cls=1)
    apply_svm(training_x, training_y, cls=3)
    apply_svm(training_x, training_y, cls=5)
    apply_svm(training_x, training_y, cls=7)
    apply_svm(training_x, training_y, cls=9)


def question_4():
    E_in1, classifier1, support_vector_count_0 = apply_svm(training_x, training_y, cls=0)
    E_in2, classifier2, support_vector_count_1 = apply_svm(training_x, training_y, cls=1)
    print('Difference between the number of support vectors of cls_0 and cls_1 = %d ' % (
            support_vector_count_0 - support_vector_count_1))


def apply_svd_classifier_list(C_list, Q=2):
    # get training data and update according to classifiers 1 and 5
    train_x = training_data[:, 1:]
    train_y = training_data[:, 0]
    i = np.logical_or(train_y == 1, train_y == 5)
    train_x = train_x[i, :]
    train_y = update_y(train_y[i], 1)

    # get test data and update according to classifiers 1 and 5
    test_x = test_data[:, 1:]
    test_y = test_data[:, 0]
    i = np.logical_or(test_y == 1, test_y == 5)
    test_x = test_x[i, :]
    test_y = update_y(test_y[i], 1)

    # create lists
    E_ins = []
    E_outs = []
    support_vectors = []

    # for each C in list
    for C in C_list:
        # apply svm
        E_in, classifier, support = apply_svm(train_x, train_y, cls=-1, C=C, Q=Q)

        # predict test data and calculate E_out
        y_predicted = classifier.predict(test_x)
        E_out = np.sum(test_y * y_predicted < 0) / (1.0 * test_y.size)

        # append values to lists
        E_ins.append(E_in)
        support_vectors.append(support)
        E_outs.append(E_out)

    # print results
    print('\nC\t\tE_in\tE_out\tSupport_vector_count')
    for i in range(len(C_list)):
        print("%.5f\t%.5f\t%.5f\t%d" % (C_list[i], E_ins[i], E_outs[i], support_vectors[i]))
    print()


def question_5():
    # apply svd with selected classifier list
    apply_svd_classifier_list([0.001, 0.01, 0.1, 1], Q=2)


def question_6():
    apply_svd_classifier_list([0.0001, 0.001, 0.01, 0.1, 1], Q=2)
    apply_svd_classifier_list([0.0001, 0.001, 0.01, 0.1, 1], Q=5)


def apply_cross_validation(x, y, C=0.01, Q=2, n_folds=10):
    kfold = KFold(n_splits=n_folds, shuffle=True)

    # create lists
    E_cvs = []
    E_ins = []
    support_vectors = []

    i = 0
    for train, test in kfold.split(x):
        i += 1
        train_x, train_y = x[train], y[train]
        test_x, test_y = x[test], y[test]
        # print('Fold: %d, training_size: %d, test_size: %d' % (i, len(train), len(test)))

        # apply svm and generate values
        E_in, classifier, support_vector = apply_svm(train_x, train_y, cls=-1, C=C, Q=Q, log=False)
        y_predict = classifier.predict(test_x)
        E_cv = np.sum(test_y * y_predict < 0) / (1.0 * y_predict.size)

        # append values to lists
        E_cvs.append(E_cv)
        E_ins.append(E_in)
        support_vectors.append(support_vector)
    return (sum(E_cvs) / len(E_cvs)), (sum(E_ins) / len(E_ins)), (sum(support_vectors) / len(support_vectors))


def question_7_8():
    C_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
    runs = 100

    # get training data and update according to classifiers 1 and 5
    train_x = training_data[:, 1:]
    train_y = training_data[:, 0]
    i = np.logical_or(train_y == 1, train_y == 5)
    train_x = train_x[i, :]
    train_y = update_y(train_y[i], 1)

    hits = np.zeros(len(C_list))
    E_cross_validations = np.empty((runs, len(C_list)))

    for i in range(runs):
        for j in range(len(C_list)):
            C = C_list[j]
            E_cv_mean, E_in_mean, sv_mean = apply_cross_validation(train_x, train_y, C=C, Q=2, n_folds=10)
            E_cross_validations[i, j] = float("{0:.4f}".format(E_cv_mean))
        index = np.argmin(E_cross_validations[i, :])
        hits[index] += 1
        print('Iteration %d, min support vector: %.4f' % (i, C_list[int(index)]))

    mean_cv = np.mean(E_cross_validations, 0).tolist()

    # print results
    print('C\t\tHits\tMean')
    for i in range(len(C_list)):
        print("%.4f\t%d\t\t%.4f" % (C_list[i], hits[i], mean_cv[i]))

    print('Mean E_cv: %.4f' % (np.mean(mean_cv)))


def question_9_10():
    # get training data and update according to classifiers 1 and 5
    train_x = training_data[:, 1:]
    train_y = training_data[:, 0]
    i = np.logical_or(train_y == 1, train_y == 5)
    train_x = train_x[i, :]
    train_y = update_y(train_y[i], 1)

    # get test data and update according to classifiers 1 and 5
    test_x = test_data[:, 1:]
    test_y = test_data[:, 0]
    i = np.logical_or(test_y == 1, test_y == 5)
    test_x = test_x[i, :]
    test_y = update_y(test_y[i], 1)

    C_list = [0.01, 1, 100, 10000, 1000000]

    # create lists
    E_ins = []
    E_outs = []

    for C in C_list:
        # run SVC with kernel rbf
        classifier = svm.SVC(kernel='rbf', C=C, shrinking=False, gamma=1, coef0=1)
        classifier.fit(train_x, train_y)

        # calculate E_in
        y_predict = classifier.predict(train_x)
        E_in = np.sum(0 > train_y * y_predict) / (1.0 * train_y.size)

        # calculate E_out
        y_test_predict = classifier.predict(test_x)
        E_out = np.sum(0 > test_y * y_test_predict) / (1.0 * test_y.size)

        # append values to lists
        E_ins.append(E_in)
        E_outs.append(E_out)

    print('\t\tC\tE_in\tE_out')
    for i in range(len(C_list)):
        print("%10.2f\t%.4f\t%.4f" % (C_list[i], E_ins[i], E_outs[i]))

def main():
    # run questions
    print('\nQuestion 2')
    question_2()
    print('\nQuestion 3')
    question_3()
    print('\nQuestion 4')
    question_4()
    print('\nQuestion 5')
    question_5()
    print('\nQuestion 6')
    question_6()
    print('\nQuestion 7-8')
    question_7_8()
    print('\nQuestion 9-10')
    question_9_10()


if __name__ == '__main__':
    main()
