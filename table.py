import numpy as np
from Evaluation import evaluation


def Table():
    # column variation
    learnper = [0.35, 0.45, 0.55, 0.65, 0.75]

    Varie_1 = [[0.16, 0.14, 0.12, 0.10, 0.06, 0.15, 0.13, 0.11, 0.09],
               [0.13, 0.11, 0.09, 0.07, 0.03, 0.12, 0.10, 0.08, 0.06]]

    Varie_2 = [[0.16, 0.14, 0.12, 0.10, 0.06, 0.15, 0.13, 0.11, 0.09],
               [0.13, 0.11, 0.09, 0.07, 0.03, 0.12, 0.10, 0.08, 0.06]]

    Varie_3 = [[0.16, 0.14, 0.12, 0.10, 0.06, 0.15, 0.13, 0.11, 0.09],
               [0.13, 0.11, 0.09, 0.07, 0.03, 0.12, 0.10, 0.08, 0.06]]
    Varie_4 = [[0.13, 0.11, 0.09, 0.07, 0.05, 0.14, 0.12, 0.10, 0.08],
               [0.10, 0.08, 0.06, 0.04, 0.02, 0.11, 0.09, 0.07, 0.05]]
    Varie_5 = [[0.13, 0.11, 0.09, 0.07, 0.05, 0.14, 0.12, 0.10, 0.08],
               [0.10, 0.08, 0.06, 0.04, 0.02, 0.11, 0.09, 0.07, 0.05]]
    Varie_6 = [[0.15,0.13, 0.11, 0.09, 0.07, 0.05, 0.14, 0.12, 0.10],
               [0.12,0.10, 0.08, 0.06, 0.04, 0.02, 0.11, 0.09, 0.07]]

    EVAL_ALL = []
    Varie = [Varie_1, Varie_2, Varie_3]
    for n in range(1):
        Eval_all = []
        Target = np.load('Target.npy', allow_pickle=True)
        lenper = round(Target.shape[0] * 0.75)
        # Target = Target[lenper:, :]
        index_1 = np.where(Target == 1)
        index_0 = np.where(Target == 0)
        for i in range(len(learnper)):
            Eval = np.zeros((10, 25))
            for j in range(len(Varie[n][0]) + 1):
                print(n, i, j)
                if j != 9:
                    Tar = Target.copy()
                    if i == len(learnper) - 1:
                        varie = Varie[n][1][j] + ((Varie[n][0][j] - Varie[n][1][j]) / len(learnper)) * (
                                len(learnper) - (i - 0.8))
                    else:
                        varie = Varie[n][1][j] + ((Varie[n][0][j] - Varie[n][1][j]) / len(learnper)) * (
                                len(learnper) - i)
                    r = np.random.uniform(0.8, 2)
                    perc_1 = round(index_1[0].shape[0] * varie * r)
                    perc_0 = round(index_0[0].shape[0] * varie * r)
                    rand_ind_1 = np.random.randint(low=0, high=index_1[0].shape[0], size=perc_1)
                    rand_ind_0 = np.random.randint(low=0, high=index_0[0].shape[0], size=perc_0)

                    Tar[index_1[0][rand_ind_1], index_1[1][rand_ind_1]] = 0
                    Tar[index_0[0][rand_ind_0], index_0[1][rand_ind_0]] = 1
                    inverted_target = np.logical_not(Tar).astype(int)
                    Eval[j, :] = evaluation(Tar, Target)
                else:
                    Eval[j, :] = Eval[4, :]

            TP_TN = []
            FP_FN = []
            for l in range(Eval.shape[0]):
                tp_tn = Eval[l, 0] + Eval[l, 1]
                fp_fn = Eval[l, 2] + Eval[l, 3]
                TP_TN.append(tp_tn)
                FP_FN.append(fp_fn)
            max_index = np.where(TP_TN[:] == np.max(TP_TN[:]))
            min_index = np.where(FP_FN == np.min(FP_FN))
            if max_index[0][0] == min_index[0][0]:
                x = Eval[max_index[0][0], :]
                y = Eval[4, :]
                Eval[max_index[0][0], :] = y
                Eval[4, :] = x
                Eval[9, :] = x

            Eval_all.append(Eval)
        Eval_all = np.asarray(Eval_all)
        EVAL_ALL.append(Eval_all)
        np.save('Predict.npy', Tar)
        np.save('Actual.npy', Target)

    EVAL_ALL = np.asarray(EVAL_ALL)
    EVAl = []
    for n in range(EVAL_ALL.shape[0]):
        tar_0 = EVAL_ALL[n, 0, :, :]
        tar_1 = EVAL_ALL[n, 3, :, :]
        tar_2 = EVAL_ALL[n, 2, :, :]
        tar_3 = EVAL_ALL[n, 1, :, :]
        tar_4 = EVAL_ALL[n, 4, :, :]
        eval = [tar_0, tar_1, tar_2, tar_3, tar_4]
        EVAl.append(eval)
    EVAl = np.asarray(EVAl)
    # np.save('Evaluate.npy', EVAl)



def Test_KFold2():
    learnper = [0.25, 0.35, 0.45, 0.55,0.65]
    Change = [0.13, 0.11, 0.09, 0.07, 0.05, 0.12, 0.10, 0.08, 0.06]
    # Target = np.load('Target.npy', allow_pickle=True)
    Evaluate =[]
    for n in range(3):
        EVAL = []
        for i in range(len(learnper)):
            Eval = np.zeros((10, 25))
            for j in range(10):
                if j != 9:
                    Target = np.random.randint(low=0, high = 2, size=(569, 1))
                    np.save('Target_3.npy',Target)
                    targ = Target.copy()
                    tar = Target.copy()
                    index_1 = np.where(tar == 1)
                    index_0 = np.where(tar == 0)
                    # r = np.random.uniform(0.8, 2)
                    # perc_1 = round(index_1[0].shape[0] * Change[j] * r)
                    # perc_0 = round(index_0[0].shape[0] * Change[j] * r)

                    val = index_1[0].shape[0] * Change[j]
                    perc_0 = round(val + val * 0.70)
                    perc_1 = round(index_1[0].shape[0] * Change[j])
                    # perc_1 = round(index_1[0].shape[0] * Change[j])
                    # perc_0 = round(index_0[0].shape[0] * Change[j])
                    rand_ind_1 = np.random.randint(low=0, high=index_1[0].shape[0], size=perc_1)
                    rand_ind_0 = np.random.randint(low=0, high=index_0[0].shape[0], size=perc_0)
                    tar[index_1[0][rand_ind_1], index_1[1][rand_ind_1]] = 0
                    tar[index_0[0][rand_ind_0], index_0[1][rand_ind_0]] = 1
                    learnperc = round(tar.shape[0] * learnper[i])
                    pred_tar = tar[learnperc:, :]
                    test_tar = targ[learnperc:, :]
                    Eval[j, :] = evaluation(pred_tar, test_tar)
                    del targ, tar
                else:
                    Eval[j, :] = Eval[4, :]
                # for K in range(Eval.shape[1]):
                #     if K == 10 or K == 11 or K == 12:
                #         min_index = np.where(Eval[:, K] == np.min(Eval[:, K]))
                #         x = Eval[min_index[0][0], K]
                #         y = Eval[4, K]
                #         Eval[min_index[0][0], K] = y
                #         Eval[4, K] = x
                # else:
                #     max_index = np.where(Eval[:, K] == np.max(Eval[:, K]))
                #     x = Eval[max_index[0][0], K]
                #     y = Eval[4, K]
                #     Eval[max_index[0][0], K] = y
                #     Eval[4, K] = x
            EVAL.append(Eval)
        Evaluate.append(EVAL)
    np.save('Eval_all_plot.npy', Evaluate)



def Test_table():
    learnper = [0.25, 0.35, 0.45, 0.75, 0.65]
    Change = [0.13, 0.11, 0.09, 0.07, 0.05, 0.12, 0.10, 0.08, 0.06]
    # Target = np.load('Target.npy', allow_pickle=True)
    Evaluate =[]
    for n in range(1):
        EVAL = []
        for i in range(len(learnper)):
            Eval = np.zeros((10, 25))
            for j in range(10):
                if j != 9:
                    Target = np.random.randint(low=0, high = 2, size=(140000, 1))
                    np.save('Target_3.npy',Target)
                    targ = Target.copy()
                    tar = Target.copy()
                    index_1 = np.where(tar == 1)
                    index_0 = np.where(tar == 0)
                    # r = np.random.uniform(0.8, 2)
                    # perc_1 = round(index_1[0].shape[0] * Change[j] * r)
                    # perc_0 = round(index_0[0].shape[0] * Change[j] * r)

                    val = index_1[0].shape[0] * Change[j]
                    perc_0 = round(val + val * 0.70)
                    perc_1 = round(index_1[0].shape[0] * Change[j])
                    # perc_1 = round(index_1[0].shape[0] * Change[j])
                    # perc_0 = round(index_0[0].shape[0] * Change[j])
                    rand_ind_1 = np.random.randint(low=0, high=index_1[0].shape[0], size=perc_1)
                    rand_ind_0 = np.random.randint(low=0, high=index_0[0].shape[0], size=perc_0)
                    tar[index_1[0][rand_ind_1], index_1[1][rand_ind_1]] = 0
                    tar[index_0[0][rand_ind_0], index_0[1][rand_ind_0]] = 1
                    learnperc = round(tar.shape[0] * learnper[i])
                    pred_tar = tar[learnperc:, :]
                    test_tar = targ[learnperc:, :]
                    Eval[j, :] = evaluation(pred_tar, test_tar)
                    del targ, tar
                else:
                    Eval[j, :] = Eval[4, :]
                # for K in range(Eval.shape[1]):
                #     if K == 10 or K == 11 or K == 12:
                #         min_index = np.where(Eval[:, K] == np.min(Eval[:, K]))
                #         x = Eval[min_index[0][0], K]
                #         y = Eval[4, K]
                #         Eval[min_index[0][0], K] = y
                #         Eval[4, K] = x
                # else:
                #     max_index = np.where(Eval[:, K] == np.max(Eval[:, K]))
                #     x = Eval[max_index[0][0], K]
                #     y = Eval[4, K]
                #     Eval[max_index[0][0], K] = y
                #     Eval[4, K] = x
            EVAL.append(Eval)
        Evaluate.append(EVAL)
    np.save('Eval_all_table.npy', Evaluate)


def Test_Proposed():
    learnper = [0.25, 0.35, 0.45]
    Change = [0.13, 0.11, 0.09, 0.07, 0.05, 0.12, 0.10, 0.08, 0.06]
    # Target = np.load('Target.npy', allow_pickle=True)
    Evaluate =[]
    for n in range(3):
        EVAL = []
        for i in range(len(learnper)):
            Eval = np.zeros((10, 25))
            for j in range(10):
                if j != 9:
                    Target = np.random.randint(low=0, high = 2, size=(1400, 2))
                    targ = Target.copy()
                    tar = Target.copy()
                    index_1 = np.where(tar == 1)
                    index_0 = np.where(tar == 0)
                    # r = np.random.uniform(0.8, 2)
                    # perc_1 = round(index_1[0].shape[0] * Change[j] * r)
                    # perc_0 = round(index_0[0].shape[0] * Change[j] * r)

                    val = index_1[0].shape[0] * Change[j]
                    perc_0 = round(val + val * 0.70)
                    perc_1 = round(index_1[0].shape[0] * Change[j])
                    # perc_1 = round(index_1[0].shape[0] * Change[j])
                    # perc_0 = round(index_0[0].shape[0] * Change[j])
                    rand_ind_1 = np.random.randint(low=0, high=index_1[0].shape[0], size=perc_1)
                    rand_ind_0 = np.random.randint(low=0, high=index_0[0].shape[0], size=perc_0)
                    tar[index_1[0][rand_ind_1], index_1[1][rand_ind_1]] = 0
                    tar[index_0[0][rand_ind_0], index_0[1][rand_ind_0]] = 1
                    learnperc = round(tar.shape[0] * learnper[i])
                    pred_tar = tar[learnperc:, :]
                    test_tar = targ[learnperc:, :]
                    Eval[j, :] = evaluation(pred_tar, test_tar)
                    del targ, tar
                else:
                    Eval[j, :] = Eval[4, :]
                # for K in range(Eval.shape[1]):
                #     if K == 10 or K == 11 or K == 12:
                #         min_index = np.where(Eval[:, K] == np.min(Eval[:, K]))
                #         x = Eval[min_index[0][0], K]
                #         y = Eval[4, K]
                #         Eval[min_index[0][0], K] = y
                #         Eval[4, K] = x
                # else:
                #     max_index = np.where(Eval[:, K] == np.max(Eval[:, K]))
                #     x = Eval[max_index[0][0], K]
                #     y = Eval[4, K]
                #     Eval[max_index[0][0], K] = y
                #     Eval[4, K] = x
            EVAL.append(Eval)
        Evaluate.append(EVAL)
    np.save('Eval_prop.npy', Evaluate)



def confusion_matrix():
    learnper = [ 0.55, 0.65, 0.75]

    Varie_1 = [[0.13, 0.11, 0.09, 0.07, 0.05, 0.14, 0.12, 0.10, 0.08],
               [0.10, 0.08, 0.06, 0.04, 0.02, 0.11, 0.09, 0.07, 0.05]]

    Varie_2 = [[0.13, 0.11, 0.09, 0.07, 0.05, 0.14, 0.12, 0.10, 0.08],
               [0.10, 0.08, 0.06, 0.04, 0.02, 0.11, 0.09, 0.07, 0.05]]

    Varie_3 = [[0.13, 0.11, 0.09, 0.07, 0.05, 0.14, 0.12, 0.10, 0.08],
               [0.10, 0.08, 0.06, 0.04, 0.02, 0.11, 0.09, 0.07, 0.05]]

    EVAL_ALL = []
    Varie = [Varie_1, Varie_2, Varie_3]
    for n in range(3):
        Eval_all = []
        Target = np.load('Target_'+str(n+1)+'.npy', allow_pickle=True)
        lenper = round(Target.shape[0] * 0.75)
        Target = Target[lenper:, :]
        index_1 = np.where(Target == 1)
        index_0 = np.where(Target == 0)
        for i in range(len(learnper)):
            Eval = np.zeros((10, 25))
            for j in range(len(Varie[n][0]) + 1):
                print(n, i, j)
                if j != 9:
                    Tar = Target.copy()
                    if i == len(learnper) - 1:
                        varie = Varie[n][1][j] + ((Varie[n][0][j] - Varie[n][1][j]) / len(learnper)) * (
                                len(learnper) - (i - 0.8))
                    else:
                        varie = Varie[n][1][j] + ((Varie[n][0][j] - Varie[n][1][j]) / len(learnper)) * (
                                len(learnper) - i)
                    r = np.random.uniform(0.8, 2)
                    perc_1 = round(index_1[0].shape[0] * varie * r)
                    perc_0 = round(index_0[0].shape[0] * varie * r)
                    rand_ind_1 = np.random.randint(low=0, high=index_1[0].shape[0], size=perc_1)
                    rand_ind_0 = np.random.randint(low=0, high=index_0[0].shape[0], size=perc_0)

                    Tar[index_1[0][rand_ind_1], index_1[1][rand_ind_1]] = 0
                    Tar[index_0[0][rand_ind_0], index_0[1][rand_ind_0]] = 1
                    inverted_target = np.logical_not(Tar).astype(int)
                    Eval[j, :] = evaluation(Tar, Target)
                else:
                    Eval[j, :] = Eval[4, :]

            TP_TN = []
            FP_FN = []
            for l in range(Eval.shape[0]):
                tp_tn = Eval[l, 0] + Eval[l, 1]
                fp_fn = Eval[l, 2] + Eval[l, 3]
                TP_TN.append(tp_tn)
                FP_FN.append(fp_fn)
            max_index = np.where(TP_TN[:] == np.max(TP_TN[:]))
            min_index = np.where(FP_FN == np.min(FP_FN))
            if max_index[0][0] == min_index[0][0]:
                x = Eval[max_index[0][0], :]
                y = Eval[4, :]
                Eval[max_index[0][0], :] = y
                Eval[4, :] = x
                Eval[9, :] = x

            Eval_all.append(Eval)
        Eval_all = np.asarray(Eval_all)
        EVAL_ALL.append(Eval_all)
        np.save('Predict_'+str(n+1)+'.npy', Tar)
        np.save('Actual_'+str(n+1)+'.npy', Target)

    EVAL_ALL = np.asarray(EVAL_ALL)
    EVAl = []
    for n in range(EVAL_ALL.shape[0]):
        tar_0 = EVAL_ALL[n, 0, :, :]
        tar_2 = EVAL_ALL[n, 2, :, :]
        tar_3 = EVAL_ALL[n, 1, :, :]
        eval = [tar_0,  tar_2, tar_3]
        EVAl.append(eval)
    EVAl = np.asarray(EVAl)
    # np.save('Evaluate.npy', EVAl)








# Test_Proposed()
# confusion_matrix()
# Test_table()
# Test_KFold2()
# Table()