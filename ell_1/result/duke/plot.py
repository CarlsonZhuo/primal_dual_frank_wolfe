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
    opt = 0.0009432
    vs_time = True
    time_trim = 10

    from matplotlib.pyplot import figure
    figure(num=None, figsize=(6, 5), dpi=100, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': 11})

    assert(len(sys.argv) == 2)
    if sys.argv[1] == 'all':
        iter_loss1, iter_time1 = parse_file('FW')
        iter_loss2, iter_time2 = parse_file('APG')
        iter_loss3, iter_time3 = parse_file('STORC')
        iter_loss4, iter_time4 = parse_file('pdFW')
        loss1 = iter_loss1 - opt
        loss2 = iter_loss2 - opt
        loss3 = iter_loss3 - opt
        loss4 = iter_loss4 - opt
        if vs_time:
            time1 = np.cumsum(iter_time1)
            time2 = np.cumsum(iter_time2)
            time3 = np.cumsum(iter_time3)
            time4 = np.cumsum(iter_time4)
            time1, loss1 = trim_by_time(time1, loss1, time_trim)
            time2, loss2 = trim_by_time(time2, loss2, time_trim)
            time3, loss3 = trim_by_time(time3, loss3, time_trim)
            time4, loss4 = trim_by_time(time4, loss4, time_trim)
            plt.semilogy(time1, loss1, label='FW')
            plt.semilogy(time2, loss2, label='Acc PG')
            plt.semilogy(time3, loss3, label='STORC')
            plt.semilogy(time4, loss4, label='Primal Dual FW')
            plt.xlabel('Running Time (seconds)')
            plt.ylabel('Relative Primal Objective')
        else:
            plt.semilogy(loss1, label='FW')
            plt.semilogy(loss2, label='Acc PG')
            plt.semilogy(loss3, label='STORC')
            plt.semilogy(loss4, label='Primal Dual FW')
        plt.legend(loc='lower right')
        plt.show()
    else:
        iter_loss, iter_time = parse_file(sys.argv[1])
        iter_time = np.cumsum(iter_time)
        iter_loss = iter_loss - min(iter_loss)
        plt.semilogy(iter_time, iter_loss)
        plt.show()

    

if __name__ == '__main__':
    main()
