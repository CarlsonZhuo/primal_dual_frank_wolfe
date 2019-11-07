import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys

def parse_file(file_name):
    with open(file_name) as dataFile:
        iter_loss = dataFile.readline()
        iter_time = dataFile.readline()
    iter_loss = iter_loss.split(', ')
    iter_time = iter_time.split(', ')
    iter_loss = iter_loss[:-1]
    iter_time = iter_time[:-1]
    iter_loss = np.array(list(map(float, iter_loss)))
    iter_time = np.array(list(map(float, iter_time)))
    return iter_loss, iter_time


def trim_by_time(time, loss, time_trim):
    if time_trim < 0:
        return time, loss
    if np.max(time) < time_trim:
        return time, loss

    idx = np.argmax(time > time_trim)
    time = time[:idx]
    loss = loss[:idx]
    return time, loss


def main():
    STRECH = True
    # ITER = True
    # STRECH = False

    opt1 = 0.399700083544505
    opt2 = 0.454154488151499
    opt3 = 0.479011853830459
    time_trim = 1
    title = 'TODO'


    from matplotlib.pyplot import figure
    figure(num=None, figsize=(8, 7), dpi=100, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': 15})

    if sys.argv[1] == 'all':
        iter_loss1, iter_time1 = parse_file('pdFW1000')
        iter_loss2, iter_time2 = parse_file('pdFW3000')
        iter_loss3, iter_time3 = parse_file('pdFW9000')
        loss1 = iter_loss1 - opt1
        loss2 = iter_loss2 - opt2
        loss3 = iter_loss3 - opt3
        if STRECH:
            time_trim = 1
            factor1 = 203 * (1000 + 1000)
            factor2 = 117 * (3000 + 3000)
            factor3 = 101 * (9000 + 9000) 
            # time1 = np.cumsum(iter_time1) / 1
            # time2 = np.cumsum(iter_time2) / 2.15
            # time3 = np.cumsum(iter_time3) / 5.65
            time1 = np.cumsum(iter_time1) / factor1 * factor2
            time2 = np.cumsum(iter_time2) / factor2 * factor2
            time3 = np.cumsum(iter_time3) / factor3 * factor2
            time1, loss1 = trim_by_time(time1, loss1, time_trim)
            time2, loss2 = trim_by_time(time2, loss2, time_trim)
            time3, loss3 = trim_by_time(time3, loss3, time_trim)
            plt.semilogy(time1, loss1, 'o-', label='dim = 1000')
            plt.semilogy(time2, loss2, 'o-', label='dim = 3000')
            plt.semilogy(time3, loss3, 'o-', label='dim = 9000')
            plt.xlabel('TODO')
            plt.ylabel('Relative Primal Objective')
        elif ITER:
            time_trim = 3
            time1 = np.cumsum(iter_time1)
            time2 = np.cumsum(iter_time2)
            time3 = np.cumsum(iter_time3)
            time1, loss1 = trim_by_time(time1, loss1, time_trim)
            time2, loss2 = trim_by_time(time2, loss2, time_trim)
            time3, loss3 = trim_by_time(time3, loss3, time_trim)
            plt.semilogy(loss1, 'o-', label='dim = 1000')
            plt.semilogy(loss2, 'o-', label='dim = 3000')
            plt.semilogy(loss3, 'o-', label='dim = 9000')
            plt.xlabel('Number of Iterations')
            plt.ylabel('Relative Primal Objective')
        else:
            time_trim = 2
            time1 = np.cumsum(iter_time1)
            time2 = np.cumsum(iter_time2)
            time3 = np.cumsum(iter_time3)
            time1, loss1 = trim_by_time(time1, loss1, time_trim)
            time2, loss2 = trim_by_time(time2, loss2, time_trim)
            time3, loss3 = trim_by_time(time3, loss3, time_trim)
            plt.semilogy(time1, loss1, 'o-', label='dim = 1000')
            plt.semilogy(time2, loss2, 'o-', label='dim = 3000')
            plt.semilogy(time3, loss3, 'o-', label='dim = 9000')
            plt.xlabel('Running Time (seconds)')
            plt.ylabel('Relative Primal Objective')
        plt.legend(loc='lower right')
        plt.title(title)
        plt.show()
    else:
        iter_loss, iter_time = parse_file(sys.argv[1])
        iter_time = np.cumsum(iter_time)
        iter_loss = iter_loss - min(iter_loss)
        plt.semilogy(iter_time, iter_loss)
        plt.show()

    

if __name__ == '__main__':
    main()
